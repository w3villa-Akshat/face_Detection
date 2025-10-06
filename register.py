import cv2
import argparse
import time
from db import init_db, add_embedding
from utils import detect_faces_gray, crop_face, get_embedding_from_face, save_face_image
from faiss_index import FaissIndexManager


def register(name, capture_count=5):
    init_db()
    idx_mgr = FaissIndexManager()
    idx_mgr.build_index()

    cap = cv2.VideoCapture(0)
    collected = 0
    print("Press 'c' to capture a face image when you're ready. Collect different angles/poses.")
    while collected < capture_count:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam")
            break
        faces = detect_faces_gray(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"Captured: {collected}/{capture_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow('Register - press c to capture, q to quit', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None
        
        elif k == ord('c'):
            if len(faces) == 0:
                print("No face detected. Try again.")
                continue
            # take the largest face
            faces_sorted = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
            face_rect = faces_sorted[0]
            face_img = crop_face(frame, face_rect)
            save_face_image(face_img, name, collected+1)
            
            emb = get_embedding_from_face(face_img)
            if emb is None:
                print("Failed to compute embedding; try a clearer face capture.")
                continue
            
            db_id = add_embedding(name, emb)
            idx_mgr.add(db_id, name, emb)
            
            collected += 1
            print(f"Captured image {collected}/{capture_count}")
            print(emb)
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    time.sleep(1)
    cv2.destroyAllWindows()
    print('Registration finished')
    return name


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--name', required=True, help='Name of the person to register')
#     args = parser.parse_args()
#     register(args.name)