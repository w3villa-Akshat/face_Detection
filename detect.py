from collections import Counter
import time
import cv2
from utils import detect_faces_gray, crop_face, get_embedding_from_face
from faiss_index import FaissIndexManager

THRESHOLD = 0.5  # adjust based on your embedding distances

def recognize_face(embedding, idx_mgr):
    
    results = idx_mgr.search(embedding, k=5)  # top 5 matches
    print("Search results:", results)
    if not results:
        return "Unknown"

    # Filter results by distance threshold
    filtered = [(db_id, name, dist) for db_id, name, dist in results if dist < THRESHOLD]
    if not filtered:
        return "Unknown"

    # Count votes per name
    names = [name for (_, name, _) in filtered]
    vote_count = Counter(names)

    # Get most common name
    candidate, count = vote_count.most_common(1)[0]

    # Require at least 3 votes out of 5 to confirm
    if count >= 3:
        return candidate
    return "Unknown"


def run_detection():
    idx_mgr = FaissIndexManager()
    idx_mgr.build_index()
    cap = cv2.VideoCapture(0)
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces_gray(frame)

        for (x, y, w, h) in faces:
            face_img = crop_face(frame, (x, y, w, h))
            emb = get_embedding_from_face(face_img)
            name = "Unknown"
            
            if emb is not None:
                # Use majority voting recognition
                name = recognize_face(emb, idx_mgr)

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # FPS display
        cv2.putText(frame, f"FPS: {int(1.0 / (time.time() - fps_time + 1e-6))}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        fps_time = time.time()

        cv2.imshow('Face Detection', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_detection()