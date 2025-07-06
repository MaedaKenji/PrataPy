import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

# --- 1. Inisialisasi ONNX Runtime -------------------------------------------
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']   # urutan fallback
session = ort.InferenceSession('cnn_model.onnx', providers=providers)

# Nama tensor I/O (tergantung ekspor, tapi biasanya 'input' & 'output')
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"ONNX providers aktif: {session.get_providers()}")

# --- 2. Definisi transform sama persis dgn training -------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                              # (C,H,W) & skala 0‑1
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # sama dgn training
                         std=[0.229, 0.224, 0.225])
])

# --- 3. Fungsi soft‑max util -------------------------------------------------
def softmax(x, axis=None):
    x_exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return x_exp / x_exp.sum(axis=axis, keepdims=True)

# --- 4. Fungsi prediksi single image ----------------------------------------
def predict_onnx(image_path, class_names):
    # 4a. Baca & praproses
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)          # shape (1,3,224,224) torch
    np_input = tensor.numpy().astype(np.float32)   # ke numpy FP32

    # 4b. Inferensi
    scores = session.run([output_name], {input_name: np_input})[0]  # shape (1,N)
    probs  = softmax(scores, axis=1)[0]          # (N,)

    # 4c. Ambil prediksi & confidence
    pred_idx   = probs.argmax()
    confidence = float(probs[pred_idx])
    return class_names[pred_idx], confidence

# --- 5. Contoh penggunaan ----------------------------------------------------
class_names = ['birch', 'elm']
img_path = '/home/ubuntu/CODE/PRATAPY/dataset-classification/bagus/birch/tree9.jpg'

pred, conf = predict_onnx(img_path, class_names)
print(f"Predicted: {pred}  |  Confidence: {conf*100:.2f}%")
