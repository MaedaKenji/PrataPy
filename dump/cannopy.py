import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Detect trees and calculate canopy area using YOLOv11 and GSD')
    parser.add_argument('--weights', type=str, default='/home/agus/Code/PRATAPY/runs/detect/treetopTrain/weights/best.pt', help='model weights path')
    parser.add_argument('--image', type=str ,default='/home/agus/Code/PRATAPY/Tree-Top-View.v1i.yolov11/valid/images/2018_TEAK_3_322000_4103000_image_50_jpeg.rf.0149482a9912f199128cd0c77f4913be.jpg' ,help='input image path')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--gsd', type=float, default='0.2', help='Ground Sample Distance in meters/pixel')
    
    return parser.parse_args()

def load_model(weights_path):
    """Load YOLOv11 model"""
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model - YOLOv11 specific
    try:
        # Load my model
        model = torch.hub.load('ultralytics/yolov11', 'custom', path=weights_path, force_reload=True)
        
    except Exception as e:
        print(f"Error loading with torch.hub: {e}")
        try:
            # Alternative loading method
            from models.experimental import attempt_load
            model = attempt_load(weights_path, device=device)
        except:
            # Fallback to YOLO class if it's a local installation
            from ultralytics import YOLO
            model = YOLO(weights_path)
            
    
    model.to(device)
    return model

def detect_trees(model, image_path, conf_thres, iou_thres, img_size):
    """Perform tree detection on the input image"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded")
    
    # Perform inference
    results = model(image_path, imgsz=img_size)
    
    # Process results based on the return type
    tree_detections = []
    
    # Handle different result formats based on YOLOv11 implementation
    try:
        # Try newer YOLOv11 format (list of Results objects)
        if isinstance(results, list):
            # Convert the results to pandas DataFrame format manually
            result = results[0]  # Get the first result
            
            # Access predictions - check if it has the boxes attribute
            if hasattr(result, 'boxes'):
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else box.xywh[0].tolist()
                    conf = box.conf[0].item() if hasattr(box, 'conf') else 0.0
                    cls = int(box.cls[0].item()) if hasattr(box, 'cls') else 0
                    
                    # Get class name from names dictionary if available
                    class_name = result.names[cls] if hasattr(result, 'names') else f"class_{cls}"
                    
                    if conf >= conf_thres:
                        tree_detections.append({
                            'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                            'confidence': conf, 'class': cls, 'name': class_name
                        })
            else:
                # Try accessing xyxy format
                for det in result:
                    if len(det):
                        for *xyxy, conf, cls in det:
                            if conf >= conf_thres:
                                x1, y1, x2, y2 = map(int, xyxy)
                                cls = int(cls)
                                class_name = result.names[cls] if hasattr(result, 'names') else f"class_{cls}"
                                tree_detections.append({
                                    'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                                    'confidence': conf, 'class': cls, 'name': class_name
                                })
        
        # Traditional pandas format
        else:
            detections = results.pandas().xyxy[0]
            # Filter for tree class
            tree_detections = detections[detections['name'] == 'tree'].to_dict('records')
            
            # If no tree class exists, try to find the most likely tree-related class
            if len(tree_detections) == 0:
                print("No specific 'tree' class found. Looking for related classes...")
                for class_name in detections['name'].unique():
                    if any(tree_term in class_name.lower() for tree_term in ['tree', 'plant', 'forest', 'vegetation']):
                        tree_detections = detections[detections['name'] == class_name].to_dict('records')
                        print(f"Using '{class_name}' as tree class.")
                        break
    
    except Exception as e:
        print(f"Error processing results: {e}")
        # Try a fallback method for newer YOLOv11 versions
        print("Attempting fallback detection method...")
        try:
            # Process raw output from model
            for r in results:
                for box in r.boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    # Get class name
                    class_name = model.names[cls] if hasattr(model, 'names') else f"class_{cls}"
                    
                    # Check if it's a tree or related class
                    is_tree = (class_name.lower() == 'tree' or 
                              any(tree_term in class_name.lower() 
                                  for tree_term in ['tree', 'plant', 'forest', 'vegetation']))
                    
                    if is_tree and conf >= conf_thres:
                        tree_detections.append({
                            'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                            'confidence': conf, 'class': cls, 'name': class_name
                        })
        except Exception as e2:
            print(f"Fallback method failed: {e2}")
            raise ValueError(f"Could not process detection results: {e}, {e2}")
    
    # Convert to pandas DataFrame for consistent handling
    tree_detections_df = pd.DataFrame(tree_detections)
    
    print(f"Detected {len(tree_detections_df)} tree instances")
    
    return tree_detections_df, img

def calculate_canopy_area(tree_detections, image, gsd):
    """Calculate tree canopy area using the GSD"""
    original_image = image.copy()
    mask = np.zeros_like(image[:, :, 0])
    
    total_area = 0
    total_agb = 0
    total_carbon = 0
    total_co2 = 0
    
    # Create a list to store tree data for pandas DataFrame
    tree_data = []
    
    for idx, detection in enumerate(tree_detections.iterrows(), 1):
        _, detection = detection
        # Extract coordinates
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        
        # Draw rectangle and tree number on original image
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Calculate pixels in the tree canopy
        # Option 1: Use bounding box area
        bbox_area_pixels = (x2 - x1) * (y2 - y1)
        
        # Option 2: Create a more precise mask if needed
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  # Fill the rectangle
        
        # Convert pixel area to real-world area using GSD
        area_square_meters = bbox_area_pixels * (gsd ** 2)
        total_area += area_square_meters
        
        # Calculate biomass and carbon values
        # AGB = 0.25 × A^1.2 (where A is the canopy area)
        agb = 0.25 * (area_square_meters ** 1.2)
        
        # Carbon = 0.47 × AGB
        carbon = 0.47 * agb
        
        # CO2 = 3.67 × Carbon
        co2 = 3.67 * carbon
        
        # Update totals
        total_agb += agb
        total_carbon += carbon
        total_co2 += co2
        
        # Add tree number and info on the image
        tree_label = f"#{idx}"
        cv2.putText(original_image, tree_label, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Add text with area information
        text = f"{area_square_meters:.2f} m2"
        cv2.putText(original_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add tree data to list for pandas DataFrame
        tree_data.append({
            'Tree': f"#{idx}",
            'Area_m2': round(area_square_meters, 2),
            'AGB_kg': round(agb, 2),
            'Carbon_kg': round(carbon, 2),
            'CO2_kg': round(co2, 2)
        })
    
    # Option 2: Calculate area from mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, contours, -1, 255, -1)
    mask_area_pixels = np.sum(contour_mask == 255)
    mask_area_square_meters = mask_area_pixels * (gsd ** 2)
    
    # Create pandas DataFrame from tree data
    df = pd.DataFrame(tree_data)
    
    # Add a row for totals
    totals = {
        'Tree': 'Total',
        'Area_m2': round(total_area, 2),
        'AGB_kg': round(total_agb, 2),
        'Carbon_kg': round(total_carbon, 2),
        'CO2_kg': round(total_co2, 2)
    }
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    
    # Save DataFrame to CSV
    csv_filename = 'tree_carbon_summary.csv'
    df.to_csv(csv_filename, index=False)
    
    # Print DataFrame to console
    print("\nTree Measurements Summary:")
    print(df.to_string(index=False))
    print(f"\nSaved tree measurements to {csv_filename}")
    
    # Visualization - both save and display images
    cv2.imwrite('detection_results.jpg', original_image)
    cv2.imwrite('canopy_mask.jpg', mask)
    
    # Display images using imshow
    cv2.imshow('Detection Results', original_image)
    cv2.imshow('Canopy Mask', mask)
    
    return total_area, mask_area_square_meters, original_image, mask, total_agb, total_carbon, total_co2

def main():
    args = parse_arguments()
    
    # Load model
    print(f"Loading YOLOv11 model from {args.weights}...")
    model = load_model(args.weights)
    
    # Detect trees
    print(f"Detecting trees in {args.image}...")
    tree_detections, image = detect_trees(model, args.image, args.conf_thres, args.iou_thres, args.img_size)
    
    # Calculate canopy area
    if len(tree_detections) > 0:
        print(f"Calculating canopy area using GSD: {args.gsd} meters/pixel...")
        bbox_area, mask_area, result_image, mask, total_agb, total_carbon, total_co2 = calculate_canopy_area(tree_detections, image, args.gsd)
        
        print(f"\nResults:")
        print(f"Number of trees detected: {len(tree_detections)}")
        print(f"Total canopy area (bounding box method): {bbox_area:.2f} square meters")
        print(f"Total canopy area (mask method): {mask_area:.2f} square meters")
        print(f"Total above-ground biomass (AGB): {total_agb:.2f} kg")
        print(f"Total carbon content: {total_carbon:.2f} kg")
        print(f"Total CO2 equivalent: {total_co2:.2f} kg")
        print(f"Detection results displayed and saved as 'detection_results.jpg'")
        print(f"Canopy mask displayed and saved as 'canopy_mask.jpg'")
        print(f"Tree measurements saved to 'tree_carbon_summary.csv'")
        
        # Wait for a key press to close windows
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No trees detected in the image.")

if __name__ == "__main__":
    main()