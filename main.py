import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Detect trees, classify them, and calculate crown area using YOLOv11, ONNX CNN, and GSD')
    parser.add_argument('--weights', type=str, default='runs/detect/treetopTrain/weights/best.pt', help='YOLO model weights path')
    parser.add_argument('--cnn-model', type=str, default='cnn_model.onnx', help='ONNX CNN model path for classification')
    parser.add_argument('--image', type=str, default='dataset-harist/DATASET_GK/Foto/GK_001.JPG', help='input image path')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--gsd', type=float, default=0.0264, help='Ground Sample Distance in meters/pixel')
    parser.add_argument('--class-names', type=str, nargs='+', default=['birch', 'elm'], 
                        help='Class names for CNN classification (e.g., --class-names birch elm oak)')
    
    return parser.parse_args()

def load_yolo_model(weights_path):
    """Load YOLOv11 model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = torch.hub.load('ultralytics/yolov11', 'custom', path=weights_path, force_reload=True)
    except Exception as e:
        print(f"Error loading with torch.hub: {e}")
        try:
            from models.experimental import attempt_load
            model = attempt_load(weights_path, device=device)
        except:
            from ultralytics import YOLO
            model = YOLO(weights_path)
    
    model.to(device)
    return model

def initialize_cnn_model(model_path):
    """Initialize ONNX CNN model for classification"""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"ONNX providers active: {session.get_providers()}")
    
    # Define preprocessing transform (same as training)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return session, input_name, output_name, preprocess

def softmax(x, axis=None):
    """Softmax function for probability calculation"""
    x_exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return x_exp / x_exp.sum(axis=axis, keepdims=True)

def classify_tree_region(image_roi, session, input_name, output_name, preprocess, class_names):
    """Classify a tree region using ONNX CNN model"""
    try:
        # Convert BGR to RGB
        rgb_roi = cv2.cvtColor(image_roi, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_roi)
        
        # Preprocess
        tensor = preprocess(pil_image).unsqueeze(0)
        np_input = tensor.numpy().astype(np.float32)
        
        # Inference
        scores = session.run([output_name], {input_name: np_input})[0]
        probs = softmax(scores, axis=1)[0]
        
        # Get prediction and confidence
        pred_idx = probs.argmax()
        confidence = float(probs[pred_idx])
        
        return class_names[pred_idx], confidence
    
    except Exception as e:
        print(f"Error classifying tree: {e}")
        return "unknown", 0.0

def detect_trees(model, image_path, conf_thres, iou_thres):
    """Perform tree detection on the input image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded")
    
    results = model(image_path)
    tree_detections = []
    
    try:
        if isinstance(results, list):
            result = results[0]
            
            if hasattr(result, 'boxes'):
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else box.xywh[0].tolist()
                    conf = box.conf[0].item() if hasattr(box, 'conf') else 0.0
                    cls = int(box.cls[0].item()) if hasattr(box, 'cls') else 0
                    
                    class_name = result.names[cls] if hasattr(result, 'names') else f"class_{cls}"
                    
                    if conf >= conf_thres:
                        tree_detections.append({
                            'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                            'confidence': conf, 'class': cls, 'name': class_name
                        })
        else:
            # Handle traditional pandas format
            detections = results.pandas().xyxy[0]
            tree_detections = detections[detections['name'] == 'tree'].to_dict('records')
            
            if len(tree_detections) == 0:
                print("No specific 'tree' class found. Looking for related classes...")
                for class_name in detections['name'].unique():
                    if any(tree_term in class_name.lower() for tree_term in ['tree', 'plant', 'forest', 'vegetation']):
                        tree_detections = detections[detections['name'] == class_name].to_dict('records')
                        print(f"Using '{class_name}' as tree class.")
                        break
    
    except Exception as e:
        print(f"Error processing results: {e}")
        # Fallback method
        try:
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    class_name = model.names[cls] if hasattr(model, 'names') else f"class_{cls}"
                    
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
    
    tree_detections_df = pd.DataFrame(tree_detections)
    print(f"Detected {len(tree_detections_df)} tree instances")
    
    return tree_detections_df, img

def calculate_crown_area_with_classification(tree_detections, image, gsd, cnn_session, input_name, output_name, preprocess, class_names):
    """Calculate tree crown area and classify each tree"""
    unmodified_image = image.copy()
    original_image = image.copy()
    colored_mask = np.zeros_like(image)
    
    # Initialize variables
    total_area = 0
    total_agb = 0
    total_carbon = 0
    total_co2 = 0
    
    font_scale = 2
    tree_data = []
    
    # Process each tree detection
    for idx, detection in enumerate(tree_detections.iterrows(), 1):
        _, detection = detection
        
        # Extract coordinates
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        
        # Extract tree region for classification
        tree_roi = image[y1:y2, x1:x2]
        
        # Skip very small detections
        if tree_roi.shape[0] < 3 or tree_roi.shape[1] < 3:
            continue
        
        # Classify the tree
        tree_species, species_confidence = classify_tree_region(
            tree_roi, cnn_session, input_name, output_name, preprocess, class_names
        )
        
        # Draw rectangle for bounding box on original image
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 15)
        
        # Create individual mask for this tree
        individual_mask = np.zeros_like(image[:, :, 0])
        
        # Convert to grayscale for masking
        tree_gray = cv2.cvtColor(tree_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to separate tree crown from background
        _, tree_thresh = cv2.threshold(tree_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up mask
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
            crown_area_pixels = np.sum(individual_mask > 0)
            
            # Convert to real-world area
            crown_area_square_meters = crown_area_pixels * (gsd ** 2)
            total_area += crown_area_square_meters
            
            # Calculate biomass and carbon values based on crown area
            BM = np.exp(2.568) * (crown_area_square_meters ** 1.418)
            agb = BM
            carbon = 0.47 * agb
            co2 = 3.67 * carbon
            
            # Update totals
            total_agb += agb
            total_carbon += carbon
            total_co2 += co2
            
            # Add tree number and species info on the image
            tree_label = f"#{idx}: {tree_species}"
            cv2.putText(original_image, tree_label, (x1, y1-60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
            
            # Add confidence info
            conf_text = f"{species_confidence*100:.1f}%"
            cv2.putText(original_image, conf_text, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)
            
            # Add tree data to list for pandas DataFrame
            tree_data.append({
                'Tree': f"#{idx}",
                'Species': tree_species,
                'Species_Confidence': round(species_confidence * 100, 1),
                'Detection_Confidence': round(detection['confidence'], 3),
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
        'Species': '-',
        'Species_Confidence': '-',
        'Detection_Confidence': '-',
        'Crown_Area_m2': round(total_area, 2),
        'AGB_kg': round(total_agb, 2),
        'Carbon_kg': round(total_carbon, 2),
        'CO2_kg': round(total_co2, 2)
    }
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    
    # Save DataFrame to CSV
    output_dir = 'tree-detection-classification'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_filename = f'{output_dir}/tree_detection_classification_summary.csv'
    df.to_csv(csv_filename, index=False)
    
    # Print DataFrame to console
    print("\nTree Detection and Classification Summary:")
    print(df.to_string(index=False))
    print(f"\nSaved tree measurements to {csv_filename}")
    
    # Overlay the colored mask on original image for visualization
    alpha = 0.5
    crown_overlay = cv2.addWeighted(original_image, 1, colored_mask, alpha, 0)
    
    # Save visualization images
    cv2.imwrite(f'{output_dir}/original_image.jpg', unmodified_image)
    cv2.imwrite(f'{output_dir}/detection_classification_results.jpg', original_image)
    cv2.imwrite(f'{output_dir}/crown_mask.jpg', total_crown_mask)
    cv2.imwrite(f'{output_dir}/crown_colored_mask.jpg', colored_mask)
    cv2.imwrite(f'{output_dir}/crown_overlay.jpg', crown_overlay)
    
    return total_area, total_crown_area_square_meters, original_image, colored_mask, total_agb, total_carbon, total_co2, df

def main():
    args = parse_arguments()
    
    # Load YOLO model
    print(f"Loading YOLOv11 model from {args.weights}...")
    yolo_model = load_yolo_model(args.weights)
    
    # Initialize CNN model
    print(f"Loading CNN classification model from {args.cnn_model}...")
    cnn_session, input_name, output_name, preprocess = initialize_cnn_model(args.cnn_model)
    
    # Detect trees
    print(f"Detecting trees in {args.image}...")
    tree_detections, image = detect_trees(yolo_model, args.image, args.conf_thres, args.iou_thres)
    
    # Calculate crown area and classify trees
    if len(tree_detections) > 0:
        print(f"Calculating tree crown area and classifying species using GSD: {args.gsd} meters/pixel...")
        crown_area, total_mask_area, result_image, colored_mask, total_agb, total_carbon, total_co2, summary_df = calculate_crown_area_with_classification(
            tree_detections, image, args.gsd, cnn_session, input_name, output_name, preprocess, args.class_names
        )
        
        print(f"\nResults:")
        print(f"Number of trees detected: {len(tree_detections)}")
        print(f"Total tree crown area: {crown_area:.2f} square meters")
        print(f"Total above-ground biomass (AGB): {total_agb:.2f} kg")
        print(f"Total carbon content: {total_carbon:.2f} kg")
        print(f"Total CO2 equivalent: {total_co2:.2f} kg")
        
        # Print species distribution
        species_counts = summary_df[summary_df['Tree'] != 'Total']['Species'].value_counts()
        print(f"\nSpecies distribution:")
        for species, count in species_counts.items():
            print(f"  {species}: {count} trees")
        
        print(f"\nOutput files saved to 'tree-detection-classification/' directory:")
        print(f"- Detection and classification results: 'detection_classification_results.jpg'")
        print(f"- Crown masks: 'crown_mask.jpg' and 'crown_colored_mask.jpg'")
        print(f"- Crown overlay: 'crown_overlay.jpg'")
        print(f"- Complete summary: 'tree_detection_classification_summary.csv'")
        
        # Wait for a key press to close windows
        print("\nPress any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No trees detected in the image.")

if __name__ == "__main__":
    main()