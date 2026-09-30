from fastapi import FastAPI, File, UploadFile
import torch
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI(title="Spatial AI Vision Backend")

print("Loading Models on Cloud Backend...")
yolo_model = YOLO("yolov8n.pt")

model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    h, w, _ = frame.shape
    results = yolo_model(frame, verbose=False)[0]
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    detections = []
    
    for r in results.boxes:
        box = r.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        label = yolo_model.names[int(r.cls[0])]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        roi_depth = depth_map[max(0, cy-10):min(h, cy+10), max(0, cx-10):min(w, cx+10)]
        mean_depth = float(np.mean(roi_depth)) if roi_depth.size > 0 else 0.0

        if mean_depth > 600:
            status = "WARNING: VERY CLOSE"
            print(f"[ALERT] {label.upper()} IS VERY CLOSE! Depth: {mean_depth:.2f}")
        elif 300 <= mean_depth <= 600:
            status = "SAFE: MID DISTANCE"
        else:
            status = "SAFE: FAR AWAY"

        detections.append({
            "label": label,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "center": [int(cx), int(cy)],
            "depth_val": mean_depth,
            "status": status
        })
        
    return {"detections": detections}