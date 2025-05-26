import pyttsx3
from threading import Thread
from queue import Queue
from ultralytics import YOLO
import cv2
import numpy as np
import time

# === 1) Text-to-Speech Thread ===
def tts_worker(q):
    engine = pyttsx3.init()
    engine.setProperty('rate', 235)
    engine.setProperty('volume', 1.0)
    while True:
        label, dist, pos = q.get()              # blocks until an item is available
        dist_rounded = round(dist * 2) / 2      # nearest 0.5
        dist_str = str(int(dist_rounded)) if dist_rounded.is_integer() else f"{dist_rounded:.1f}"
        text = f"{label} is {dist_str} meters to your {pos}"
        print(f"[TTS] {text}")
        engine.say(text)
        engine.runAndWait()
        q.task_done()

tts_queue = Queue()
Thread(target=tts_worker, args=(tts_queue,), daemon=True).start()

# === 2) Load YOLOv8 Model ===
model = YOLO("yolov8n-oiv7.pt")
print("Model loaded. Classes:")
for idx, name in model.names.items():
    print(f"  {idx}: {name}")
print()

# === 3) Width Ratios & Colors ===
# Default width_ratio=1.0 if class not specified
class_info = {
    "person": {"width_ratio": 2.5,  "color": (0,255,0)},
    "car":    {"width_ratio": 0.37, "color": (0,255,255)},
    "bicycle": {"width_ratio": 2.3},
    "motorcycle": {"width_ratio": 2.4},
    "bus": {"width_ratio": 0.3},
    "traffic light": {"width_ratio": 2.95},
    "stop sign": {"width_ratio": 2.55},
    "bench": {"width_ratio": 1.6},
    "cat": {"width_ratio": 1.9},
    "dog": {"width_ratio": 1.5},
    "chair": {"width_ratio": 1.0},
    "table": {"width_ratio": 1.2},
    "laptop": {"width_ratio": 0.5},
    "book": {"width_ratio": 0.3},
    "blackboard": {"width_ratio": 2.0},
    "whiteboard": {"width_ratio": 2.0},
    "projector": {"width_ratio": 0.8},
    "backpack": {"width_ratio": 0.4},
    "bottle": {"width_ratio": 0.2},
    "cup": {"width_ratio": 0.2},
    "keyboard": {"width_ratio": 0.5},
    "mouse": {"width_ratio": 0.2},
    "pen": {"width_ratio": 0.1},
    "pencil": {"width_ratio": 0.1},
    "notebook": {"width_ratio": 0.3},
    "desk": {"width_ratio": 1.5},
    "monitor": {"width_ratio": 0.7},
    "printer": {"width_ratio": 0.6},
    "telephone": {"width_ratio": 0.3},
    "scissors": {"width_ratio": 0.2},
    "stapler": {"width_ratio": 0.2},
    "tape": {"width_ratio": 0.2},
    "calculator": {"width_ratio": 0.3},
    "glasses": {"width_ratio": 0.2},
    "clock": {"width_ratio": 0.5},
    "trash can": {"width_ratio": 0.6},
    "door": {"width_ratio": 1.0},
    "window": {"width_ratio": 1.2},
    "board eraser": {"width_ratio": 0.3},
    "marker": {"width_ratio": 0.1},
    "highlighter": {"width_ratio": 0.1},
    "ruler": {"width_ratio": 0.5},
    "paper": {"width_ratio": 0.3},
    "envelope": {"width_ratio": 0.3},
    "folder": {"width_ratio": 0.4},
    "file cabinet": {"width_ratio": 1.0},
    "projector screen": {"width_ratio": 2.0},
    "speaker": {"width_ratio": 0.6},
    "microphone": {"width_ratio": 0.2},
    "headphones": {"width_ratio": 0.3},
    "remote control": {"width_ratio": 0.3},
    "light switch": {"width_ratio": 0.1},
    "thermostat": {"width_ratio": 0.2},
    "fire extinguisher": {"width_ratio": 0.5},
    "first aid kit": {"width_ratio": 0.4},
    "hand sanitizer": {"width_ratio": 0.2},
    "tissue box": {"width_ratio": 0.3},
    "water dispenser": {"width_ratio": 0.7},
    "coffee machine": {"width_ratio": 0.6},
    "microwave": {"width_ratio": 0.6},
    "refrigerator": {"width_ratio": 0.8},
    "sink": {"width_ratio": 0.7},
    "soap dispenser": {"width_ratio": 0.2},
    "paper towel dispenser": {"width_ratio": 0.5},
    "toilet": {"width_ratio": 0.7},
    "urinal": {"width_ratio": 0.5},
    "mirror": {"width_ratio": 1.0},
    "hand dryer": {"width_ratio": 0.4},
    "trash bin": {"width_ratio": 0.6},
    "recycling bin": {"width_ratio": 0.6},
    "bulletin board": {"width_ratio": 1.5},
    "poster": {"width_ratio": 1.2},
    "painting": {"width_ratio": 1.2},
    "flag": {"width_ratio": 1.0},
    "globe": {"width_ratio": 0.5},
    "trophy": {"width_ratio": 0.4},
    "medal": {"width_ratio": 0.2},
    "certificate": {"width_ratio": 0.5},
    "award": {"width_ratio": 0.4},
    "plant": {"width_ratio": 0.6},
    "vase": {"width_ratio": 0.3},
    "curtain": {"width_ratio": 1.0},
    "blinds": {"width_ratio": 1.0},
    "fan": {"width_ratio": 0.7},
    "air conditioner": {"width_ratio": 0.8},
    "heater": {"width_ratio": 0.7},
    "radiator": {"width_ratio": 0.7},
    "lamp": {"width_ratio": 0.5},
    "chandelier": {"width_ratio": 1.0},
    "light bulb": {"width_ratio": 0.2},
    "ceiling light": {"width_ratio": 1.0},
    "floor lamp": {"width_ratio": 0.5},
    "table lamp": {"width_ratio": 0.4},
    "wall clock": {"width_ratio": 0.5},
    "alarm clock": {"width_ratio": 0.3},
    "calendar": {"width_ratio": 0.5},
    "whiteboard eraser": {"width_ratio": 0.3},
    "chalk": {"width_ratio": 0.1},
    "chalkboard": {"width_ratio": 2.0},
    "projector remote": {"width_ratio": 0.3},
    "laser pointer": {"width_ratio": 0.2},
    "presentation clicker": {"width_ratio": 0.3},
    "pointer stick": {"width_ratio": 0.5},
    "name tag": {"width_ratio": 0.2},
    "id card": {"width_ratio": 0.2},
    "badge": {"width_ratio": 0.2},
    "lanyard": {"width_ratio": 0.3},
    "clipboard": {"width_ratio": 0.4},
    "paper clip": {"width_ratio": 0.1},
    "push pin": {"width_ratio": 0.1},
    "rubber band": {"width_ratio": 0.1},
    "staple remover": {"width_ratio": 0.2},
    "binder": {"width_ratio": 0.5},
    "hole punch": {"width_ratio": 0.3},
    "paper shredder": {"width_ratio": 0.6},
    "filing cabinet": {"width_ratio": 1.0},
    "bookcase": {"width_ratio": 1.5},
    "bookshelf": {"width_ratio": 1.5},
    "magazine rack": {"width_ratio": 0.8},
    "newspaper": {"width_ratio": 0.5},
    "magazine": {"width_ratio": 0.5},
    "journal": {"width_ratio": 0.5},
    "notepad": {"width_ratio": 0.3},
    "sketchbook": {"width_ratio": 0.4},
    "drawing pad": {"width_ratio": 0.4},
    "art supplies": {"width_ratio": 0.5},
    "paintbrush": {"width_ratio": 0.2},
    "palette": {"width_ratio": 0.4},
    "easel": {"width_ratio": 1.0},
    # you can add more key/value entries here
}

# === 4) Helper Functions ===
def calculate_distance(box, frame_w, label):
    w_px = (box.xyxy[0,2] - box.xyxy[0,0]).item()
    ratio = class_info.get(label, {}).get("width_ratio", 1.0)
    w_px *= ratio
    # pinhole camera model
    return round((frame_w * 0.5) / np.tan(np.radians(70/2)) / (w_px + 1e-6), 2)

def get_position(frame_w, x1):
    third = frame_w // 3
    if x1 < third:      return "left"
    if x1 < 2*third:    return "forward"
    return "right"

def blur_person(img, box):
    x,y,_,h = box.xyxy[0].cpu().numpy().astype(int)
    w = int((box.xyxy[0,2] - box.xyxy[0,0]).item())
    top = img[y:y+int(0.08*h), x:x+w]
    img[y:y+int(0.08*h), x:x+w] = cv2.GaussianBlur(top, (15,15), 0)
    return img

# === 5) Open Camera ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

print("Starting detection. Press 'q' or ESC to exit.")

# === 6) Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False)
    res = results[0]

    print(f"Frame detections: {len(res.boxes)}")

    for box in res.boxes:
        cls_id = int(box.cls[0].item())
        label  = model.names[cls_id]
        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]

        # calculate distance for every label
        dist = calculate_distance(box, frame.shape[1], label)
        pos  = get_position(frame.shape[1], x1)

        # blur person’s top region
        if label == "person":
            frame = blur_person(frame, box)

        # choose color
        color = class_info.get(label, {}).get("color", (255,0,0))

        # draw box & label
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"{label}:{dist:.1f}m", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # if closer than threshold, highlight red and queue audio
        if dist <= 12.5:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            print(f"[DETECT] {label} at {dist:.1f}m → queueing audio ({pos})")
            tts_queue.put((label, dist, pos))

    cv2.imshow("Audio World (All Objects)", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
