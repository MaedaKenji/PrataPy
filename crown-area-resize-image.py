"""
Kesimpulan:
Ketika imgage di resize, bounding box yang dihasilkan tidak sesuai dengan ukuran asli image.
Penghitungan menjadi tidak akurat karena bounding box yang dihasilkan tidak sesuai dengan ukuran asli image.
"""


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
    parser = argparse.ArgumentParser(description='Detect trees and calculate crown area using YOLOv11 and GSD')
    parser.add_argument('--weights', type=str, default='runs/detect/treetopTrain/weights/best.pt', help='model weights path')
    parser.add_argument('--image', type=str ,default='dataset-harist/DATASET_GK/GK_001.JPG' ,help='input image path')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--gsd', type=float, default='0.0264', help='Ground Sample Distance in meters/pixel')
    
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
    img = cv2.resize(img, (img_size, img_size))  # Resize image to model input size

    # Perform inference on the resized image directly
    results = model(img, imgsz=img_size)
    
    
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

def calculate_crown_area(tree_detections, image, gsd):
    """Calculate tree crown area using masking for more accurate crown measurement"""
    # Save a copy of the completely unmodified original image
    unmodified_image = image.copy()
    
    original_image = image.copy()
    
    # Create a colored mask for visualization
    colored_mask = np.zeros_like(image)
    
    # Initialize variables
    total_area = 0
    total_agb = 0
    total_carbon = 0
    total_co2 = 0
    
    # Create a list to store tree data for pandas DataFrame
    tree_data = []
    
    # Process each tree detection
    for idx, detection in enumerate(tree_detections.iterrows(), 1):
        _, detection = detection
        # Extract coordinates
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        
        # Draw rectangle for bounding box on original image
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Create individual mask for this tree
        individual_mask = np.zeros_like(image[:, :, 0])
        
        # Extract the tree region from the image
        tree_roi = image[y1:y2, x1:x2]
        
        # Skip very small detections
        if tree_roi.shape[0] < 3 or tree_roi.shape[1] < 3:
            continue
            
        # Convert to grayscale
        tree_gray = cv2.cvtColor(tree_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to separate tree crown from background
        # Use Otsu's method to determine optimal threshold automatically
        _, tree_thresh = cv2.threshold(tree_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Optional: Apply morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        tree_thresh = cv2.morphologyEx(tree_thresh, cv2.MORPH_OPEN, kernel)
        tree_thresh = cv2.morphologyEx(tree_thresh, cv2.MORPH_CLOSE, kernel)
        
        # Place the processed ROI back into the individual mask
        individual_mask[y1:y2, x1:x2] = tree_thresh
        
        # Find contours in this tree's mask
        contours, _ = cv2.findContours(individual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If contours found, fill them in the colored mask with a unique color
        if contours:
            # Generate a unique color for this tree
            tree_color = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
            cv2.drawContours(colored_mask, contours, -1, tree_color, -1)
            
            # Calculate crown area in pixels
            crown_area_pixels = np.sum(individual_mask > 0)  # Count non-zero pixels
            
            # Convert to real-world area
            crown_area_square_meters = crown_area_pixels * (gsd ** 2)
            total_area += crown_area_square_meters
            
            # Calculate biomass and carbon values based on crown area
            # AGB = 0.25 × A^1.2 (where A is the crown area)
            agb = 0.25 * (crown_area_square_meters ** 1.2)
            
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
            
            # Add area information to the image
            text = f"{crown_area_square_meters:.2f} m2"
            cv2.putText(original_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add tree data to list for pandas DataFrame
            tree_data.append({
                'Tree': f"#{idx}",
                'Crown_Area_m2': round(crown_area_square_meters, 2),
                'AGB_kg': round(agb, 2),
                'Carbon_kg': round(carbon, 2),
                'CO2_kg': round(co2, 2)
            })
    
    # Calculate total crown area from the colored mask
    total_crown_mask = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2GRAY)
    total_crown_mask = (total_crown_mask > 0).astype(np.uint8) * 255
    total_crown_area_pixels = np.sum(total_crown_mask > 0)
    total_crown_area_square_meters = total_crown_area_pixels * (gsd ** 2)
    
    # Create pandas DataFrame from tree data
    df = pd.DataFrame(tree_data)
    
    # Add a row for totals
    totals = {
        'Tree': 'Total',
        'Crown_Area_m2': round(total_area, 2),
        'AGB_kg': round(total_agb, 2),
        'Carbon_kg': round(total_carbon, 2),
        'CO2_kg': round(total_co2, 2)
    }
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    
    # Save DataFrame to CSV
    output_dir = 'crown-area'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_filename = f'{output_dir}/tree_crown_carbon_summary.csv'
    df.to_csv(csv_filename, index=False)
    
    # Print DataFrame to console
    print("\nTree Crown Measurements Summary:")
    print(df.to_string(index=False))
    print(f"\nSaved tree measurements to {csv_filename}")
    
    # Overlay the colored mask on original image for visualization
    alpha = 0.5
    crown_overlay = cv2.addWeighted(original_image, 1, colored_mask, alpha, 0)
    
    # Visualization - save and display images
    # Ensure output directory exists
    output_dir = 'crown-area'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cv2.imwrite(f'{output_dir}/original_image.jpg', unmodified_image)
    cv2.imwrite(f'{output_dir}/detection_results.jpg', original_image)
    cv2.imwrite(f'{output_dir}/crown_mask.jpg', total_crown_mask)
    cv2.imwrite(f'{output_dir}/crown_colored_mask.jpg', colored_mask)
    cv2.imwrite(f'{output_dir}/crown_overlay.jpg', crown_overlay)
    
    # Display images using imshow
    # cv2.imshow('Original Image', unmodified_image)
    # cv2.imshow('Detection Results', original_image)
    # cv2.imshow('Crown Masks (Colored)', colored_mask)
    # cv2.imshow('Crown Overlay', crown_overlay)
    
    return total_area, total_crown_area_square_meters, original_image, colored_mask, total_agb, total_carbon, total_co2

def main():
    args = parse_arguments()
    
    # Load model
    print(f"Loading YOLOv11 model from {args.weights}...")
    model = load_model(args.weights)
    
    # Detect trees
    print(f"Detecting trees in {args.image}...")
    tree_detections, image = detect_trees(model, args.image, args.conf_thres, args.iou_thres, args.img_size)
    
    # Calculate crown area
    if len(tree_detections) > 0:
        print(f"Calculating tree crown area using GSD: {args.gsd} meters/pixel...")
        crown_area, total_mask_area, result_image, colored_mask, total_agb, total_carbon, total_co2 = calculate_crown_area(tree_detections, image, args.gsd)
        
        print(f"\nResults:")
        print(f"Number of trees detected: {len(tree_detections)}")
        print(f"Total tree crown area: {crown_area:.2f} square meters")
        print(f"Total above-ground biomass (AGB): {total_agb:.2f} kg")
        print(f"Total carbon content: {total_carbon:.2f} kg")
        print(f"Total CO2 equivalent: {total_co2:.2f} kg")
        print(f"Original image displayed and saved as 'original_image.jpg'")
        print(f"Detection results displayed and saved as 'detection_results.jpg'")
        print(f"Crown masks displayed and saved as 'crown_mask.jpg' and 'crown_colored_mask.jpg'")
        print(f"Crown overlay displayed and saved as 'crown_overlay.jpg'")
        print(f"Tree measurements saved to 'tree_crown_carbon_summary.csv'")
        
        # Wait for a key press to close windows
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No trees detected in the image.")

if __name__ == "__main__":
    main()