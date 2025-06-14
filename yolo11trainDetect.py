from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")
# model = YOLO("yolov8n.pt")


# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="Tree-Top-View.v1i.yolov11/data.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="cuda",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("Tree-Top-View.v1i.yolov11/test/images/2_jpeg.rf.ece7a8bb6a414a860b50f67840a665c1.jpg")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model