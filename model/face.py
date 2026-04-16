import cv2
import numpy as np
from deepface import DeepFace
import faiss
import pickle
import os

class FaceRecognitionSystem:
    def __init__(self, db_path="face_db"):
        self.db_path = db_path
        self.index_file = os.path.join(db_path, "faiss_index.index")
        self.mapping_file = os.path.join(db_path, "name_mapping.pkl")
        self.model_name = "Facenet" 
        self.embedding_size = 128 
        
        self.threshold = 10.0 

        # SỬ DỤNG OPENCV HAAR CASCADE BẰNG THƯ VIỆN CÓ SẴN (THAY CHO MEDIAPIPE)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Khởi tạo hoặc load Database
        if not os.path.exists(db_path):
            os.makedirs(db_path)
            
        if os.path.exists(self.index_file) and os.path.exists(self.mapping_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.mapping_file, "rb") as f:
                self.name_mapping = pickle.load(f)
            print(f"Đã load Database với {self.index.ntotal} khuôn mặt.")
        else:
            self.index = faiss.IndexFlatL2(self.embedding_size) 
            self.name_mapping = {}
            print("Khởi tạo Database mới.")

    def _get_face_embedding(self, face_img):
        """Trích xuất vector từ ảnh khuôn mặt đã crop"""
        try:
            result = DeepFace.represent(img_path=face_img, model_name=self.model_name, 
                                        enforce_detection=False, align=True)
            return np.array(result[0]["embedding"], dtype=np.float32)
        except Exception as e:
            return None

    def register_user(self, username, num_samples=100):
        """Thu thập data và lưu vào FAISS"""
        cap = cv2.VideoCapture(0)
        count = 0
        
        print(f"\n--- ĐĂNG KÝ CHO: {username} ---")
        print("Vui lòng nhìn vào camera. Xoay nhẹ đầu sang trái, phải, lên, xuống...")

        while count < num_samples:
            ret, frame = cap.read()
            if not ret: continue

            # Detect khuôn mặt bằng OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,     # Quét chi tiết hơn một chút
                minNeighbors=8,      # TĂNG LÊN 8: Yêu cầu độ chắc chắn cao hơn, lọc bỏ các nhiễu nhỏ như miệng
                minSize=(120, 120)   # TĂNG LÊN 120x120: Khuôn mặt thực tế nhìn vào webcam chắc chắn lớn hơn 120 pixel. Miệng thì nhỏ hơn nên sẽ bị loại luôn.
            )

            for (x, y, w, h) in faces:
                # Mở rộng vùng crop một chút để lấy trọn cằm và tóc
                x_start = max(0, x - 15)
                y_start = max(0, y - 25)
                x_end = min(frame.shape[1], x + w + 15)
                y_end = min(frame.shape[0], y + h + 35)

                face_crop = frame[y_start:y_end, x_start:x_end]

                if face_crop.shape[0] > 50 and face_crop.shape[1] > 50:
                    embedding = self._get_face_embedding(face_crop)
                    
                    if embedding is not None:
                        self.index.add(np.expand_dims(embedding, axis=0))
                        self.name_mapping[self.index.ntotal - 1] = username
                        count += 1
                        
                        # Vẽ khung xanh
                        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                        cv2.putText(frame, f"Da thu: {count}/{num_samples}", (x_start, y_start-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Chỉ xử lý 1 khuôn mặt trong lúc đăng ký để tránh nhiễu
                break 

            cv2.imshow("Registering User", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        faiss.write_index(self.index, self.index_file)
        with open(self.mapping_file, "wb") as f:
            pickle.dump(self.name_mapping, f)
            
        print(f"Hoàn tất đăng ký cho {username}!\n")
        cap.release()
        cv2.destroyAllWindows()

    def recognize(self):
        """Chạy nhận diện thời gian thực"""
        if self.index.ntotal == 0:
            print("Database trống! Vui lòng đăng ký user trước.")
            return

        cap = cv2.VideoCapture(0)
        print("\n--- BẮT ĐẦU NHẬN DIỆN --- (Nhấn 'q' để thoát)")

        while True:
            ret, frame = cap.read()
            if not ret: continue

            # Detect khuôn mặt bằng OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,     # Quét chi tiết hơn một chút
                minNeighbors=8,      # TĂNG LÊN 8: Yêu cầu độ chắc chắn cao hơn, lọc bỏ các nhiễu nhỏ như miệng
                minSize=(120, 120)   # TĂNG LÊN 120x120: Khuôn mặt thực tế nhìn vào webcam chắc chắn lớn hơn 120 pixel. Miệng thì nhỏ hơn nên sẽ bị loại luôn.
            )

            for (x, y, w, h) in faces:
                x_start = max(0, x - 15)
                y_start = max(0, y - 25)
                x_end = min(frame.shape[1], x + w + 15)
                y_end = min(frame.shape[0], y + h + 35)

                face_crop = frame[y_start:y_end, x_start:x_end]

                if face_crop.shape[0] > 50 and face_crop.shape[1] > 50:
                    embedding = self._get_face_embedding(face_crop)
                    
                    if embedding is not None:
                        distances, indices = self.index.search(np.expand_dims(embedding, axis=0), k=1)
                        
                        dist = distances[0][0]
                        idx = indices[0][0]

                        if dist < self.threshold:
                            name = self.name_mapping.get(idx, "Unknown")
                            color = (0, 255, 0) 
                            label = f"{name} ({dist:.1f})"
                        else:
                            name = "Unknown"
                            color = (0, 0, 255) 
                            label = f"Unknown ({dist:.1f})"

                        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)
                        cv2.putText(frame, label, (x_start, y_start - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# ================= MÃ THỰC THI =================
if __name__ == "__main__":
    app = FaceRecognitionSystem()
    
    while True:
        print("\n=== MENU ===")
        print("1. Đăng ký người mới")
        print("2. Bật nhận diện camera")
        print("3. Thoát")
        
        choice = input("Chọn chức năng (1/2/3): ")
        
        if choice == '1':
            user_name = input("Nhập tên User mới: ")
            app.register_user(user_name)
        elif choice == '2':
            app.recognize()
        elif choice == '3':
            break
        else:
            print("Lựa chọn không hợp lệ!")