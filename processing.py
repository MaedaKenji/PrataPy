from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import torch
from pathlib import Path
import pandas as pd
import base64
import io
import os
from werkzeug.utils import secure_filename
import tempfile
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(weights_path='runs/detect/treetopTrain/weights/best.pt'):
    """Load YOLOv11 model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Try to load custom model
        model = torch.hub.load('ultralytics/yolov11', 'custom', path=weights_path, force_reload=True)
    except Exception as e:
        print(f"Error loading custom model: {e}")
        try:
            # Fallback to YOLO class
            from ultralytics import YOLO
            model = YOLO(weights_path)
        except Exception as e2:
            print(f"Error loading with YOLO class: {e2}")
            # Use a pre-trained model as fallback
            model = torch.hub.load('ultralytics/yolov11', 'yolov11n.pt')
            print("Using pre-trained YOLOv11n model as fallback")
    
    model.to(device)
    return model

def detect_trees(model, image_path, conf_thres=0.25, iou_thres=0.45, img_size=640):
    """Perform tree detection on the input image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded")
    
    # Perform inference
    results = model(image_path, imgsz=img_size)
    
    tree_detections = []
    
    try:
        if isinstance(results, list):
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    class_name = result.names[cls] if hasattr(result, 'names') else f"class_{cls}"
                    
                    # Accept all detections as potential trees
                    if conf >= conf_thres:
                        tree_detections.append({
                            'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                            'confidence': conf, 'class': cls, 'name': class_name
                        })
    
    except Exception as e:
        print(f"Error processing results: {e}")
        raise ValueError(f"Could not process detection results: {e}")
    
    tree_detections_df = pd.DataFrame(tree_detections)
    print(f"Detected {len(tree_detections_df)} objects")
    
    return tree_detections_df, img

def calculate_crown_area(tree_detections, image, gsd=0.2):
    """Calculate tree crown area using masking"""
    original_image = image.copy()
    colored_mask = np.zeros_like(image)
    
    total_area = 0
    total_agb = 0
    total_carbon = 0
    total_co2 = 0
    
    tree_data = []
    
    for idx, detection in enumerate(tree_detections.iterrows(), 1):
        _, detection = detection
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        
        # Draw rectangle for bounding box
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        individual_mask = np.zeros_like(image[:, :, 0])
        tree_roi = image[y1:y2, x1:x2]
        
        if tree_roi.shape[0] < 3 or tree_roi.shape[1] < 3:
            continue
            
        tree_gray = cv2.cvtColor(tree_roi, cv2.COLOR_BGR2GRAY)
        _, tree_thresh = cv2.threshold(tree_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((3, 3), np.uint8)
        tree_thresh = cv2.morphologyEx(tree_thresh, cv2.MORPH_OPEN, kernel)
        tree_thresh = cv2.morphologyEx(tree_thresh, cv2.MORPH_CLOSE, kernel)
        
        individual_mask[y1:y2, x1:x2] = tree_thresh
        
        contours, _ = cv2.findContours(individual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            tree_color = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
            cv2.drawContours(colored_mask, contours, -1, tree_color, -1)
            
            crown_area_pixels = np.sum(individual_mask > 0)
            crown_area_square_meters = crown_area_pixels * (gsd ** 2)
            total_area += crown_area_square_meters
            
            # Calculate biomass and carbon values
            # ln(BM) = a + b × ln(ca) 
            # BM = e^(a) x ca^b
            # Biomass (BM) in kg, a = 2.568, b = 1.418, ca is the crown area in square meters.
            BM = np.exp(2.568) * (crown_area_square_meters ** 1.418)
            agb = BM
            # agb = 0.25 * (crown_area_square_meters ** 1.2)
            carbon = 0.47 * agb
            co2 = 3.67 * carbon
            
            total_agb += agb
            total_carbon += carbon
            total_co2 += co2
            
            # Add labels to image
            tree_label = f"#{idx}"
            cv2.putText(original_image, tree_label, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            text = f"{crown_area_square_meters:.2f} m2"
            cv2.putText(original_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            tree_data.append({
                'Tree': f"#{idx}",
                'Crown_Area_m2': round(crown_area_square_meters, 2),
                'AGB_kg': round(agb, 2),
                'Carbon_kg': round(carbon, 2),
                'CO2_kg': round(co2, 2)
            })
    
    # Create overlay
    alpha = 0.5
    crown_overlay = cv2.addWeighted(original_image, 1, colored_mask, alpha, 0)
    
    return {
        'total_area': round(total_area, 2),
        'total_agb': round(total_agb, 2),
        'total_carbon': round(total_carbon, 2),
        'total_co2': round(total_co2, 2),
        'tree_count': len(tree_data),
        'tree_data': tree_data,
        'result_image': original_image,
        'crown_overlay': crown_overlay,
        'colored_mask': colored_mask
    }

def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode()
    return f"data:image/jpeg;base64,{img_str}"

# Global model variable
model = None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Tree Detection API is running'})

@app.route('/load-model', methods=['POST'])
def load_model_endpoint():
    global model
    try:
        weights_path = request.json.get('weights_path', 'runs/detect/treetopTrain/weights/best.pt')
        model = load_model(weights_path)
        return jsonify({'success': True, 'message': 'Model loaded successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    global model
    
    if model is None:
        # Try to load default model
        try:
            model = load_model()
        except Exception as e:
            return jsonify({'success': False, 'error': f'Model not loaded: {str(e)}'}), 400
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File type not allowed'}), 400
    
    try:
        # Get parameters from form data
        conf_thres = float(request.form.get('conf_thres', 0.25))
        iou_thres = float(request.form.get('iou_thres', 0.45))
        img_size = int(request.form.get('img_size', 640))
        gsd = float(request.form.get('gsd', 0.2))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        tree_detections, image = detect_trees(model, filepath, conf_thres, iou_thres, img_size)
        
        if len(tree_detections) > 0:
            results = calculate_crown_area(tree_detections, image, gsd)
            
            # Convert images to base64 for web display
            results['result_image_base64'] = image_to_base64(results['result_image'])
            results['crown_overlay_base64'] = image_to_base64(results['crown_overlay'])
            results['colored_mask_base64'] = image_to_base64(results['colored_mask'])
            
            # Remove the actual image arrays to reduce response size
            del results['result_image']
            del results['crown_overlay']
            del results['colored_mask']
            
            # Save results to CSV
            df = pd.DataFrame(results['tree_data'])
            totals = {
                'Tree': 'Total',
                'Crown_Area_m2': results['total_area'],
                'AGB_kg': results['total_agb'],
                'Carbon_kg': results['total_carbon'],
                'CO2_kg': results['total_co2']
            }
            df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
            
            csv_filename = f"tree_results_{filename.split('.')[0]}.csv"
            csv_path = os.path.join(app.config['RESULTS_FOLDER'], csv_filename)
            df.to_csv(csv_path, index=False)
            
            results['csv_filename'] = csv_filename
            results['success'] = True
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify(results)
        else:
            # Clean up uploaded file
            os.remove(filepath)
            return jsonify({'success': False, 'error': 'No trees detected in the image'})
            
    except Exception as e:
        # Clean up uploaded file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/download-csv/<filename>', methods=['GET'])
def download_csv(filename):
    try:
        csv_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if os.path.exists(csv_path):
            return send_file(csv_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)