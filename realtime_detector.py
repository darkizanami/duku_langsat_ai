# realtime_detector.py

import cv2
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import time

# --- Konfigurasi ---
YOLO_MODEL_PATH = 'yolov8n.pt'
CLASSIFICATION_MODEL_PATH = 'model/duku_langsat_model.h5'
LABELS_PATH = 'model/labels.txt'
IMAGE_SIZE = (150, 150)

# --- Daftar kelas YOLO yang dianggap 'buah' atau 'makanan' ---
# Hanya objek dengan ID di daftar ini yang akan diklasifikasikan lebih lanjut.
# Objek dengan ID di luar daftar ini akan langsung dilabeli "BUKAN BUAH".
YOLO_GENERAL_FRUIT_FOOD_CLASSES_ID = [
    46, # 'banana'
    47, # 'apple'
    49, # 'orange'
    52,
    53, 
    56,
    54,
    32,
    # Tambahkan ID lain yang relevan yang Anda anggap 'buah' atau 'makanan'
    # Anda bisa melihat semua ID dengan: print(yolo_model.names) di awal skrip (setelah model dimuat)
]

# --- Muat Model ---
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"Model YOLOv8 {YOLO_MODEL_PATH} berhasil dimuat.")

    classification_model = tf.keras.models.load_model(CLASSIFICATION_MODEL_PATH)
    with open(LABELS_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print("Model klasifikasi duku/langsat dan label berhasil dimuat.")
    print("Kelas yang dikenali:", class_names)

except Exception as e:
    print(f"Error memuat model: {e}")
    print("Pastikan semua path model benar dan model tersedia.")
    print("Jika YOLOv8n.pt tidak ada, pastikan koneksi internet aktif saat pertama kali menjalankan.")
    exit()

# --- Fungsi Preprocessing Gambar (untuk klasifikasi) ---
def preprocess_image_for_classification(image_array):
    try:
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil = image_pil.resize(IMAGE_SIZE)
        image_np = np.array(image_pil) / 255.0
        image_np = np.expand_dims(image_np, axis=0)
        return image_np
    except Exception as e:
        return None

# --- Fungsi Klasifikasi ---
def classify_fruit(cropped_image_array):
    processed_image = preprocess_image_for_classification(cropped_image_array)
    if processed_image is None:
        return "Tidak Diketahui", 0.0

    predictions = classification_model.predict(processed_image, verbose=0)
    predicted_class_index = np.argmax(predictions)
    predicted_label = class_names[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index]) * 100
    return predicted_label, confidence

# --- Main Loop untuk Kamera Real-time ---
def run_realtime_detector():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera. Pastikan kamera terhubung dan tidak digunakan oleh aplikasi lain.")
        return

    window_name = 'Duku & Langsat Detector - Real-time'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Memulai deteksi real-time dari kamera... Tekan 'q' untuk keluar.")
    
    # --- DEBUGGING PRINT (Opsional, bisa dihapus atau dikomentari) ---
    # print(f"DEBUG: YOLO_GENERAL_FRUIT_FOOD_CLASSES_ID yang digunakan: {YOLO_GENERAL_FRUIT_FOOD_CLASSES_ID}")
    # print(f"DEBUG: yolo_model.names (semua kelas YOLO): {yolo_model.names}")
    # --- AKHIR DEBUGGING PRINT ---

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Gagal mengambil frame dari kamera. Keluar.")
            break

        results = yolo_model.predict(source=frame, conf=0.5, verbose=False)

        for r in results:
            if not r.boxes:
                pass
            else:
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                names = r.names

                for box, conf, cls_idx in zip(boxes, confidences, classes):
                    x1, y1, x2, y2 = map(int, box)
                    yolo_label = names[int(cls_idx)]
                    
                    # --- DEBUGGING PRINT (Opsional, bisa dihapus atau dikomentari) ---
                    # print(f"DEBUG: Objek terdeteksi oleh YOLO - ID: {int(cls_idx)}, Label: {yolo_label}")
                    # --- AKHIR DEBUGGING PRINT ---

                    cropped_img = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    
                    display_text = ""
                    box_color = (0, 255, 0) # Default: Hijau

                    if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                        # Logika utama: Cek apakah objek YOLO termasuk kategori buah/makanan
                        if int(cls_idx) in YOLO_GENERAL_FRUIT_FOOD_CLASSES_ID:
                            # Jika ya, klasifikasikan lebih lanjut sebagai duku/langsat
                            fruit_label, fruit_confidence = classify_fruit(cropped_img)
                            
                            # Tentukan teks dan warna berdasarkan hasil klasifikasi duku/langsat
                            if fruit_label == "duku":
                                display_text = f"DUKU ({fruit_confidence:.1f}%)"
                                box_color = (255, 255, 0) # Kuning untuk duku
                            elif fruit_label == "langsat":
                                display_text = f"LANGSAT ({fruit_confidence:.1f}%)"
                                box_color = (0, 255, 255) # Cyan untuk langsat
                            else:
                                # Jika model klasifikasi tidak yakin atau hasilnya bukan duku/langsat
                                # Tapi objeknya terdeteksi sebagai "buah umum" oleh YOLO
                                display_text = f"BUAH: {yolo_label}"
                                box_color = (0, 165, 255) # Oranye untuk buah umum

                        else:
                            # Jika objek yang dideteksi YOLO BUKAN kategori buah/makanan
                            display_text = f"{yolo_label} | BUKAN BUAH" # Tampilkan label YOLO + BUKAN BUAH
                            box_color = (0, 0, 255) # Merah

                        # Gambar kotak di frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        
                        # Gambar latar belakang untuk teks
                        (text_width, text_height), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        text_y_pos = y1 - text_height - 10 if y1 - text_height - 10 > 0 else y1 + 5
                        cv2.rectangle(frame, (x1, text_y_pos - 5), (x1 + text_width, text_y_pos + text_height + 5), box_color, -1)
                        # Gambar teks
                        cv2.putText(frame, display_text, (x1, text_y_pos + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        frame_count += 1
        if frame_count % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = time.time()

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detektor real-time dihentikan.")

# --- Jalankan Detektor Real-time ---
if __name__ == '__main__':
    run_realtime_detector()