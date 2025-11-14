import os
import sys
import argparse
import glob
import time
import psutil

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "yolo/yolo11n_saved_model/yolo11n_float32.tflite")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)  
user_res = args.resolution
record = args.record

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model and get label names
model = YOLO(model_path, task='detect')
labels = model.names

#Estimate RAM used by model (using process memory)
process = psutil.Process(os.getpid())
ram_used_bytes = process.memory_info().rss
ram_used_mb = ram_used_bytes / (1024 * 1024)

model_file_size_bytes = os.path.getsize(model_path)
model_file_size_mb = model_file_size_bytes / (1024 * 1024)

print(f"Model loaded. Approx. RAM usage by process: {ram_used_mb:.2f} MB")
print(f"Model file size: {model_file_size_mb:.2f} MB")

# Parse input to determine if image source is a file, folder, video, or USB camera
#img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP','.webp','.WEBP']

vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    # Set up recording
    record_name = 'Elec537_demo.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []

    filelist = glob.glob(os.path.join(img_source, '**', '*'), recursive=True) #Check subfolders too

    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type in ['video','usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)

    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
inference_time_buffer = []
inference_avg_len = 20

# Begin inference loop
while True:
    t_start = time.perf_counter()

    # Load frame from image source
    if source_type in ['image','folder']:
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count += 1
    elif source_type in ['video','usb']:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('No more frames or camera disconnected. Exiting program.')
            break

    # Resize frame if needed
    if resize:
        frame = cv2.resize(frame,(resW,resH))

    # Run inference
    t_infer_start = time.perf_counter() 
    results = model(frame, verbose=False)
    t_infer_stop = time.perf_counter() 
    inference_time_ms = (t_infer_stop - t_infer_start) * 1000
    
    if len(inference_time_buffer) >= inference_avg_len:
        inference_time_buffer.pop(0)
    inference_time_buffer.append(inference_time_ms)
    inference_time_avg = np.mean(inference_time_buffer)
    
    detections = results[0].boxes
    object_count = 0
    

    # Draw detections
    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1

    # Calculate FPS  
    t_stop = time.perf_counter()
    frame_time_sec = t_stop - t_start
    frame_time_ms = frame_time_sec * 1000 
    frame_rate_calc = 1.0 / frame_time_sec

    # Append to FPS buffer and calculate average
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

    # Overlay info on frame
    if source_type in ['video','usb','picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f} ({frame_time_ms:.1f} ms)', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        cv2.putText(frame, f'Avg Inference: {inference_time_avg:.1f} ms', (10,60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.putText(frame, f'Objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.imshow('YOLO detection results',frame)


    if source_type == 'folder':
        output_dir = 'inference_results'
        os.makedirs(output_dir, exist_ok=True)
        # Construct output filename
        base_name = os.path.basename(img_filename)
        save_path = os.path.join(output_dir, base_name)

        print(f"Saved result: {save_path}")
        output_root = 'inference_results'
        os.makedirs(output_root, exist_ok=True)

        # Preserve relative folder structure
        rel_path = os.path.relpath(img_filename, img_source)
        save_path = os.path.join(output_root, model_path, rel_path)
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        # Save annotated image
        cv2.imwrite(save_path, frame)

        # Save detections to text file
        model_name = os.path.basename(model_path)                # e.g., "yolo11n_float32.tflite"
        model_name_noext = os.path.splitext(model_name)[0]       # e.g., "yolo11n_float32"
        txt_path = os.path.join(output_root, f'results_{model_name_noext}.txt')

        # Ensure directory exists (output_root already created above)
        if not os.path.exists(txt_path):
            # Create empty file with header
            with open(txt_path, 'w') as f:
                f.write("image_path, class, confidence, xmin, ymin, xmax, ymax\n")

        # Append detection results
        with open(txt_path, 'a') as f:
            if object_count == 0:
                f.write(f"{rel_path}, no_detections\n")
            else:
                for i in range(len(detections)):
                    conf = detections[i].conf.item()
                    if conf > min_thresh:
                        xyxy = detections[i].xyxy.cpu().numpy().squeeze()
                        xmin, ymin, xmax, ymax = xyxy.astype(int)
                        classidx = int(detections[i].cls.item())
                        classname = labels[classidx]
                f.write(f"{rel_path}, {classname}, {conf:.3f}, {xmin}, {ymin}, {xmax}, {ymax}\n")

        print(f"Saved result: {save_path}")
    
    
    if record: recorder.write(frame)

    key = cv2.waitKey(5) if source_type in ['video','usb','picamera'] else cv2.waitKey(1)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite(f'capture_{model_path}.png',frame)

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f} ({np.mean([1000.0/f for f in frame_rate_buffer]):.1f} ms/frame)')
if source_type in ['video','usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()
