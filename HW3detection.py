import cv2
import numpy as np
import tensorflow as tf
import time
from pathlib import Path
import psutil
from datetime import datetime

# Default COCO label map (indexes correspond to common TF mapping with index 0 = 'N/A').
COCO_LABELS = [
    'N/A', 'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','N/A','stop sign','parking meter','bench','bird','cat','dog','horse',
    'sheep','cow','elephant','bear','zebra','giraffe','N/A','backpack','umbrella','N/A',
    'N/A','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
    'baseball glove','skateboard','surfboard','tennis racket','bottle','N/A','wine glass','cup','fork','knife',
    'spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
    'donut','cake','chair','couch','potted plant','bed','N/A','dining table','N/A','N/A',
    'toilet','N/A','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','N/A','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]


def load_labelmap_from_file(path: str):
    p = Path(path)
    if not p.exists():
        return None
    labels = []
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                labels.append(s)
    return labels


def get_label_name(class_id: int, labels=None) -> str:
    # prefer provided labels
    if labels:
        if 0 <= class_id < len(labels):
            return labels[class_id]
        # try shifted
        if 0 <= class_id-1 < len(labels):
            return labels[class_id-1]

    # fallback to COCO_LABELS
    if 0 <= class_id < len(COCO_LABELS):
        name = COCO_LABELS[class_id]
        if name != 'N/A':
            return name
    # try shifted
    if 0 <= class_id+1 < len(COCO_LABELS):
        name = COCO_LABELS[class_id+1]
        if name != 'N/A':
            return name
    return f'ID {class_id}'

def nms(boxes, scores, classes, iou_threshold=0.5, score_threshold=0.05):
    import numpy as np
    from collections import defaultdict

    final_boxes = []
    final_scores = []
    final_classes = []

    # Filter low score boxes
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]

    # Perform NMS per class
    unique_classes = np.unique(classes)
    for c in unique_classes:
        idxs = np.where(classes == c)[0]
        class_boxes = boxes[idxs]
        class_scores = scores[idxs]

        x1 = class_boxes[:,1]
        y1 = class_boxes[:,0]
        x2 = class_boxes[:,3]
        y2 = class_boxes[:,2]

        areas = (x2 - x1) * (y2 - y1)
        order = class_scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            order = order[1:][iou < iou_threshold]

        final_boxes.extend(class_boxes[keep])
        final_scores.extend(class_scores[keep])
        final_classes.extend([c]*len(keep))

    return np.array(final_boxes), np.array(final_scores), np.array(final_classes)


def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T','P']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"


def model_disk_size_bytes(model_dir: str) -> int:
    """Return total size in bytes of a model directory (recursively)."""
    total = 0
    p = Path(model_dir)
    if not p.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    for f in p.rglob('*'):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def get_ram_usage_bytes() -> int:
    """Return current process RSS in bytes."""
    proc = psutil.Process()
    return proc.memory_info().rss

# Load TFLite model
MODEL_PATH = "efficientdet_d0.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Print model size and current RAM usage
try:
    p_model = Path(MODEL_PATH)
    if p_model.exists():
        print(f"Model file: {p_model} size: {sizeof_fmt(p_model.stat().st_size)}")
except Exception:
    pass
print(f"Process RSS: {sizeof_fmt(get_ram_usage_bytes())}")

# inference timing storage
inference_times = []
MAX_TIMES = 200

# Open camera
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

print("✅ Camera feed started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Preprocess
    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)
    if input_details[0]['dtype'] == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    else:
        input_data = input_data.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    # measure inference time
    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()
    inference_ms = (t1 - t0) * 1000.0
    inference_times.append(inference_ms)
    if len(inference_times) > MAX_TIMES:
        inference_times.pop(0)
    avg_inference = sum(inference_times) / len(inference_times)
    timestamp = datetime.now().strftime('%H:%M:%S')

    # Extract boxes and class scores
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]     # shape [76725,4]
    class_scores = interpreter.get_tensor(output_details[3]['index'])[0]  # shape [76725,90]

    # Get max score per box and its class
    max_scores = np.max(class_scores, axis=1)       # shape [76725]
    classes = np.argmax(class_scores, axis=1)      # shape [76725]

    # Draw boxes with high confidence
    imH, imW, _ = frame.shape

    # Prefilter: keep only boxes with score >= SCORE_THRESH and at most TOP_K highest scores
    SCORE_THRESH = 0.2
    TOP_K = 1000

    # Mask low scores first
    high_idx = np.where(max_scores >= SCORE_THRESH)[0]
    if high_idx.size == 0:
        # nothing to draw this frame
        cv2.imshow("TFLite Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Select top-K among high-scoring boxes to reduce NMS work
    if high_idx.size > TOP_K:
        # get indices of the top-K scores within the high_idx set
        top_within = np.argsort(max_scores[high_idx])[-TOP_K:]
        selected_idx = high_idx[top_within]
    else:
        selected_idx = high_idx

    boxes_sel = boxes[selected_idx]
    scores_sel = max_scores[selected_idx]
    classes_sel = classes[selected_idx]

    boxes_nms, scores_nms, classes_nms = nms(boxes_sel, scores_sel, classes_sel)

    # try to load optional label file (one label per line)
    user_labels = load_labelmap_from_file('coco_labels.txt') or load_labelmap_from_file('labelmap.txt')

    for i in range(len(scores_nms)):
        ymin, xmin, ymax, xmax = boxes_nms[i]
        x1, y1, x2, y2 = int(xmin*imW), int(ymin*imH), int(xmax*imW), int(ymax*imH)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cid = int(classes_nms[i])
        name = get_label_name(cid, labels=user_labels)
        label = f"{name}: {scores_nms[i]:.2f}"
        cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)


    # Overlay inference timing and timestamp
    fps_text = f"Inf: {inference_ms:.1f} ms (avg {avg_inference:.1f} ms)"
    time_text = f"Time: {timestamp}"
    mem_text = f"RSS: {sizeof_fmt(get_ram_usage_bytes())}"
    cv2.putText(frame, fps_text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, time_text, (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1)
    cv2.putText(frame, mem_text, (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1)

    cv2.imshow("TFLite Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
