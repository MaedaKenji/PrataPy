import cv2

# Path to input video
input_video = "4K Video of Highway Traffic! [KBsqQez-O4w].mp4"  # Change this to your video path
output_video = "resized_video.mp4"  # Output video name

# Read the video
cap = cv2.VideoCapture(input_video)

# Get original video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (640, 320))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame to 300x300
    resized_frame = cv2.resize(frame, (640, 320))
    
    # Write the resized frame
    out.write(resized_frame)

# Release resources
cap.release()
out.release()
print("Video processing completed!")

