# train_and_run_lbph.py
import os
import cv2
import numpy as np

DATA_DIR = "faces"
MODEL_FILE = "lbph_face.yml"
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_images_and_labels(data_dir):
    images, labels, names = [], [], {}
    label_id = 0
    for person in sorted(os.listdir(data_dir)):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir): continue
        names[label_id] = person
        for fname in os.listdir(person_dir):
            path = os.path.join(person_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            faces = FACE_CASCADE.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
            if len(faces)==0:
                # use full image if detection fails
                faces = [(0,0,img.shape[1],img.shape[0])]
            for (x,y,w,h) in faces:
                face = cv2.resize(img[y:y+h, x:x+w], (200,200))
                images.append(face)
                labels.append(label_id)
        label_id += 1
    return images, np.array(labels), names

def train():
    print("Loading images...")
    images, labels, names = load_images_and_labels(DATA_DIR)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, labels)
    model.write(MODEL_FILE)
    # save names mapping
    np.save("names.npy", names)
    print("Trained. Saved:", MODEL_FILE)

def run_realtime():
    if not os.path.exists(MODEL_FILE):
        raise SystemExit("Train first (run script with --train).")
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(MODEL_FILE)
    names = np.load("names.npy", allow_pickle=True).item()
    cam = cv2.VideoCapture(0)
    while True:
        ok, frame = cam.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for (x,y,w,h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], (200,200))
            label, conf = model.predict(face)
            text = f"{names.get(label,'unknown')}"
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,text,(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.imshow("LBPH Face", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    cam.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if "--train" in sys.argv:
        train()
    else:
        run_realtime()
