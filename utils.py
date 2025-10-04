import cv2
import numpy as np
from deepface import DeepFace
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '-1'  # suppress TensorFlow warnings
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# cascade xml path
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def detect_faces_gray(frame):
    """Return list of (x,y,w,h) for faces in frame (BGR)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    return faces


def crop_face(frame, rect):
    x, y, w, h = rect
    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_frame, x + w)
    y2 = min(h_frame, y + h)
    return frame[y1:y2, x1:x2]


def get_embedding_from_face(face_bgr):

    # Convert BGR → RGB
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    # Get embedding from DeepFace
    rep = DeepFace.represent(
        img_path=rgb,
        model_name='OpenFace',         
        detector_backend='opencv', 
        enforce_detection=False
    )
    # Extract vector
    vec = None
    if isinstance(rep, list) and rep:
        rep_item = rep[0]
        if isinstance(rep_item, dict) and 'embedding' in rep_item:
            vec = np.array(rep_item['embedding'], dtype=np.float16)
        else:
            vec = np.array(rep_item, dtype=np.float16)
    elif isinstance(rep, dict) and 'embedding' in rep:
        vec = np.array(rep['embedding'], dtype=np.float16)

    if vec is None:
        return None

    # L2-normalize to make the magnitude 1
    norm = np.linalg.norm(vec)
    if norm > 0 and norm <=1:
        vec = vec / norm
    return vec

def ensure_data_dir():
    os.makedirs('data', exist_ok=True)


def save_face_image(face_img, name, idx):
    ensure_data_dir()
    filename = os.path.join('data', f"{name}_{idx}.jpg")
    cv2.imwrite(filename, face_img)
    return filename






# def get_embedding_from_face(face_bgr):
    # """Use DeepFace to get embedding from a BGR image (numpy array)."""
    # # DeepFace.represent accepts RGB images or paths; convert to RGB
    # rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    # # DeepFace expects either a file or np array; enforce_detection False to avoid raising if small face
    # rep = DeepFace.represent(img_path = rgb, model_name='Facenet', detector_backend='opencv', enforce_detection=False)
    # # DeepFace.represent returns a list with a dict or just vector depending on versions. Normalize.
    # if isinstance(rep, list) and rep:
    #     rep_item = rep[0]
    #     if isinstance(rep_item, dict) and 'embedding' in rep_item:
    #         vec = np.array(rep_item['embedding'], dtype=np.float32)
    #         return vec
    #     # sometimes it's just a list
    #     return np.array(rep_item, dtype=np.float32)
    # if isinstance(rep, dict) and 'embedding' in rep:
    #     return np.array(rep['embedding'], dtype=np.float32)
    # return None