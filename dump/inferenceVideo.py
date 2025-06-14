from ultralytics import YOLO
import cv2
import time

# Load model yang sudah dilatih
MODEL_PATH = "yolo11n.pt"  # Ganti dengan path ke model terlatih
model = YOLO(MODEL_PATH)

# Path ke video yang akan diprediksi
VIDEO_PATH = "4K Video of Highway Traffic! [KBsqQez-O4w].mp4"  # Ganti dengan path ke video yang diunduh

# Buka video menggunakan OpenCV
cap = cv2.VideoCapture(VIDEO_PATH)

# Periksa apakah video berhasil dibuka
if not cap.isOpened():
    print("Error: Tidak dapat membuka video.")
    exit()


# Inisialisasi variabel untuk menghitung FPS
fps = 0
frame_count = 0
start_time = time.time()

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video selesai atau tidak dapat membaca frame.")
        break  # Keluar dari loop jika video selesai

    # Lakukan prediksi pada frame saat ini
    results = model.predict(
        source=frame,  # Input frame
        imgsz=640,     # Ukuran input gambar (harus sama dengan saat training)
        conf=0.5,      # Threshold confidence (misalnya 0.5)
    )

    # Gambar bounding box pada frame
    for result in results:
        boxes = result.boxes.xyxy  # Koordinat bounding box (format: [x1, y1, x2, y2])
        classes = result.boxes.cls  # Indeks kelas
        confidences = result.boxes.conf  # Confidence score

        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)  # Konversi koordinat ke integer
            label = f"Class {int(cls)}: {conf:.2f}"  # Label untuk bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Gambar kotak
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Tambahkan teks
            count += 1
            
    # Hitung FPS setiap 1 detik
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        
    # Tampilkan FPS di frame
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame yang telah dimodifikasi
    cv2.imshow("Live Detection", frame)

    # Tunggu 1 ms dan keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan sumber daya
cap.release()
cv2.destroyAllWindows()

print(f"Jumlah deteksi total: {count}")