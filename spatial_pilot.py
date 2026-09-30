import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import torch
from ultralytics import YOLO

device = "cpu"
print(f"Running Spatial AI on device: {device}")

print("Loading YOLOv8 & MiDaS Depth Models...")
yolo_model = YOLO("yolov8n.pt")

depth_model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", depth_model_type)
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Webcam disconnected.")
        break

    h, w, _ = frame.shape

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.numpy()
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

    results = yolo_model(frame, stream=True, verbose=False, device="cpu")

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = yolo_model.names[cls]

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            ymin, ymax = max(0, cy - 1), min(h, cy + 2)
            xmin, xmax = max(0, cx - 1), min(w, cx + 2)
            
            grid = depth_map[ymin:ymax, xmin:xmax]
            obj_depth = np.median(grid) if grid.size > 0 else depth_map[cy, cx]

            if obj_depth > 500:
                status, color = "WARNING: VERY CLOSE!", (0, 0, 255)
            elif obj_depth > 250:
                status, color = "MID RANGE", (0, 255, 255)
            else:
                status, color = "FAR AWAY", (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1) # Center point visualization
            cv2.putText(frame, f"{label} | {status}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Spatial AI - Object Detection", frame)
    cv2.imshow("Depth Map Feed", depth_colormap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()