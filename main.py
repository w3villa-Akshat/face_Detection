# --- file: main.py ---
"""
Simple CLI that lets you pick register or detect.
"""
import argparse
import register
import detect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['register', 'detect'], help='Mode: register or detect')
    parser.add_argument('--name', help='Name (required for register)')
    args = parser.parse_args()
    if args.mode == 'register':
        if not args.name:
            print('Please provide --name for registration')
            return
        register.register(args.name)
    else:
        detect.run_detection()

if __name__ == '__main__':
    main()


# --- file: README.md ---
"""
Requirements:
- Python 3.8+
- pip install deepface opencv-python faiss-cpu numpy

Notes & tuning:
- The code uses OpenCV Haar cascade for face detection and DeepFace(Facenet) for embeddings.
- The L2 distance threshold (THRESHOLD in detect.py) needs to be tuned for your environment and model.
- To register: python main.py register --name "Alice"
- To detect: python main.py detect

"""
