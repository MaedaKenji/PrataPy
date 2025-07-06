import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
import pandas as pd
import os

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Detect trees and save individual tree images for crown shape classification')
    parser.add_argument('--weights', type=str, default='runs/detect/treetopTrain/weights/best.pt', help='model weights path')
    parser.add_argument('--input-folder', type=str, default='/home/ubuntu/CODE/PRATAPY/dataset-harist/DATASET_GK/Foto', help='input folder containing images')
    parser.add_argument('--image', type=str, default='', help='single image path (optional, overrides input-folder)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--output-dir', type=str, default='singletree', help='output directory for individual tree images')
    parser.add_argument('--padding', type=int, default=20, help='padding around detected tree bounding box')
    parser.add_argument('--min-size', type=int, default=50, help='minimum size for tree detection (width or height)')
    # for one file use: python tree_extraction.py --image /path/to/single/image.jpg
    # with custom parameters: python tree_extraction.py --input-folder /path/to/your/images --output-dir singletree --padding 30 --min-size 60
    return parser.parse_args()

def load_model(weights_path):
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

def get_image_files(folder_path):
    """Get all image files from the specified folder"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF']
    image_files = []
    
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise ValueError(f"Folder {folder_path} does not exist")
    
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f'*{ext}'))
    
    return sorted(image_files)


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
            # Fallback for pandas format
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
    
    tree_detections_df = pd.DataFrame(tree_detections)
    print(f"Detected {len(tree_detections_df)} tree instances")
    
    return tree_detections_df, img


def process_single_image(model, image_path, conf_thres, iou_thres, output_dir, padding, min_size, global_tree_count):
    """Process a single image and return updated global tree count"""
    print(f"\nProcessing: {image_path}")
    
    try:
        # Detect trees
        tree_detections, image = detect_trees(model, str(image_path), conf_thres, iou_thres)
        
        if len(tree_detections) == 0:
            print(f"No trees detected in {image_path.name}")
            return global_tree_count, []
        
        # Extract individual trees
        image_name = image_path.stem
        valid_tree_count, tree_info = extract_individual_trees(
            tree_detections, image, output_dir, padding, min_size, global_tree_count, image_name
        )
        
        print(f"Extracted {valid_tree_count} trees from {image_path.name}")
        return global_tree_count + valid_tree_count, tree_info
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return global_tree_count, []
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
            # Fallback for pandas format
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
    
    tree_detections_df = pd.DataFrame(tree_detections)
    print(f"Detected {len(tree_detections_df)} tree instances")
    
    return tree_detections_df, img

def extract_individual_trees(tree_detections, image, output_dir, padding, min_size, global_tree_count=0, image_name=""):
    """Extract individual tree images and save them"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create visualization image
    result_image = image.copy()
    
    # List to store tree information
    tree_info = []
    
    # Process each tree detection
    valid_tree_count = 0
    
    for idx, (_, detection) in enumerate(tree_detections.iterrows()):
        # Extract coordinates
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        
        # Calculate width and height
        width = x2 - x1
        height = y2 - y1
        
        # Skip very small detections
        if width < min_size or height < min_size:
            print(f"Skipping tree {idx + 1} from {image_name}: too small ({width}x{height})")
            continue
        
        # Add padding to bounding box
        img_height, img_width = image.shape[:2]
        
        # Calculate padded coordinates
        x1_padded = max(0, x1 - padding)
        y1_padded = max(0, y1 - padding)
        x2_padded = min(img_width, x2 + padding)
        y2_padded = min(img_height, y2 + padding)
        
        # Extract tree region with padding
        tree_crop = image[y1_padded:y2_padded, x1_padded:x2_padded]
        
        # Skip if crop is too small
        if tree_crop.shape[0] < 10 or tree_crop.shape[1] < 10:
            print(f"Skipping tree {idx + 1} from {image_name}: crop too small after padding")
            continue
        
        valid_tree_count += 1
        global_tree_id = global_tree_count + valid_tree_count
        
        # Save individual tree image
        tree_filename = f"{output_dir}/tree{global_tree_id}.jpg"
        cv2.imwrite(tree_filename, tree_crop)
        
        # Draw bounding box on result image
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Add tree number label
        label = f"T{global_tree_id}"
        cv2.putText(result_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Store tree information
        tree_info.append({
            'Tree_ID': global_tree_id,
            'Source_Image': image_name,
            'Filename': f"tree{global_tree_id}.jpg",
            'Original_BBox': f"({x1}, {y1}, {x2}, {y2})",
            'Padded_BBox': f"({x1_padded}, {y1_padded}, {x2_padded}, {y2_padded})",
            'Width': width,
            'Height': height,
            'Confidence': round(detection['confidence'], 3),
            'Class': detection['name'],
            'Crown_Shape': 'TBD'  # To be determined by classification
        })
        
        print(f"Saved tree {global_tree_id} from {image_name}: {tree_filename} (size: {tree_crop.shape[1]}x{tree_crop.shape[0]})")
    
    # Save detection results image for this source image
    if valid_tree_count > 0:
        result_filename = f"{output_dir}/detection_{image_name}.jpg"
        cv2.imwrite(result_filename, result_image)
        print(f"Saved detection results to {result_filename}")
    
    return valid_tree_count, tree_info

def create_classification_template(output_dir, tree_count):
    """Create a template CSV for manual crown shape classification"""
    
    # Crown shape categories
    crown_shapes = [
        'pyramidal',      # Triangular/conical shape
        'cylindrical',    # Column-like shape
        'ovoid',         # Egg-shaped
        'half-ellipsoid', # Half-ellipse shape
        'spherical'      # Round/circular shape
    ]
    
    # Create template data
    template_data = []
    for i in range(1, tree_count + 1):
        template_data.append({
            'Tree_ID': i,
            'Filename': f"tree{i}.jpg",
            'Crown_Shape': '',  # Empty for manual classification
            'Notes': ''
        })
    
    # Save template
    template_filename = f"{output_dir}/crown_shape_classification_template.csv"
    template_df = pd.DataFrame(template_data)
    template_df.to_csv(template_filename, index=False)
    
    # Create instructions file
    instructions = f"""
Crown Shape Classification Instructions
=====================================

Please classify each tree crown shape by editing the crown_shape_classification_template.csv file.

Available crown shapes:
1. pyramidal - Triangular/conical shape (wider at base, pointed at top)
2. cylindrical - Column-like shape (similar width throughout height)
3. ovoid - Egg-shaped (wider in middle, tapered at both ends)
4. half-ellipsoid - Half-ellipse shape (rounded dome-like)
5. spherical - Round/circular shape (ball-like)

Instructions:
1. Open each tree image in the {output_dir} folder
2. Observe the crown shape
3. Enter the appropriate crown shape in the 'Crown_Shape' column
4. Add any additional notes in the 'Notes' column
5. Save the file when complete

Tree images: tree1.jpg to tree{tree_count}.jpg
Template file: {template_filename}
"""
    
    instructions_filename = f"{output_dir}/classification_instructions.txt"
    with open(instructions_filename, 'w') as f:
        f.write(instructions)
    
    print(f"Created classification template: {template_filename}")
    print(f"Created instructions file: {instructions_filename}")

def main():
    args = parse_arguments()
    
    # Load model
    print(f"Loading YOLOv11 model from {args.weights}...")
    model = load_model(args.weights)
    
    # Determine input source
    all_tree_info = []
    total_tree_count = 0
    
    if args.image:
        # Process single image
        print(f"Processing single image: {args.image}")
        image_files = [Path(args.image)]
    else:
        # Process all images in folder
        print(f"Processing all images in folder: {args.input_folder}")
        image_files = get_image_files(args.input_folder)
        
        if not image_files:
            print(f"No image files found in {args.input_folder}")
            return
        
        print(f"Found {len(image_files)} image files")
    
    # Process each image
    for image_path in image_files:
        try:
            updated_count, tree_info = process_single_image(
                model, image_path, args.conf_thres, args.iou_thres, 
                args.output_dir, args.padding, args.min_size, total_tree_count
            )
            
            all_tree_info.extend(tree_info)
            total_tree_count = updated_count
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Save consolidated results
    if all_tree_info:
        # Create consolidated CSV
        df = pd.DataFrame(all_tree_info)
        csv_filename = f"{args.output_dir}/all_trees_list.csv"
        df.to_csv(csv_filename, index=False)
        
        # Create classification template
        create_classification_template(args.output_dir, total_tree_count)
        
        # Create summary report
        create_summary_report(args.output_dir, len(image_files), total_tree_count, all_tree_info)
        
        print(f"\n{'='*50}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"Images processed: {len(image_files)}")
        print(f"Total trees extracted: {total_tree_count}")
        print(f"Output directory: {args.output_dir}/")
        print(f"Individual tree images: tree1.jpg to tree{total_tree_count}.jpg")
        print(f"Consolidated tree list: {csv_filename}")
        print(f"Classification template: {args.output_dir}/crown_shape_classification_template.csv")
        print(f"Processing summary: {args.output_dir}/processing_summary.txt")
    else:
        print("No trees were extracted from any images.")

def create_summary_report(output_dir, total_images, total_trees, tree_info):
    """Create a summary report of the processing"""
    
    # Count trees per source image
    source_counts = {}
    for tree in tree_info:
        source = tree['Source_Image']
        source_counts[source] = source_counts.get(source, 0) + 1
    
    # Create summary report
    summary = f"""
Tree Extraction Processing Summary
================================

Processing Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Overview:
- Total images processed: {total_images}
- Total trees extracted: {total_trees}
- Average trees per image: {total_trees/total_images:.2f}

Trees per source image:
"""
    
    for source, count in sorted(source_counts.items()):
        summary += f"- {source}: {count} trees\n"
    
    summary += f"""
Output Files:
- Individual tree images: tree1.jpg to tree{total_trees}.jpg
- Detection results: detection_[image_name].jpg for each source image
- Consolidated tree list: all_trees_list.csv
- Classification template: crown_shape_classification_template.csv
- Processing instructions: classification_instructions.txt

Next Steps:
1. Review individual tree images in the output directory
2. Use the classification template to manually classify crown shapes
3. Train a machine learning model for automated classification
"""
    
    summary_filename = f"{output_dir}/processing_summary.txt"
    with open(summary_filename, 'w') as f:
        f.write(summary)
    
    print(f"Created processing summary: {summary_filename}")

if __name__ == "__main__":
    main()