import cv2
import requests
import os
import threading
import time
from gtts import gTTS
import pygame

pygame.mixer.quit()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)

SERVER_URL = "http://127.0.0.1:8000/process_frame"

is_speaking = False
last_speech_time = 0
repeat_count = 0

MAX_REPETITIONS = 4
GAP_BETWEEN_SPEAKS = 5.0   
LONG_PAUSE_COOLDOWN = 30.0  

def play_audio_file(text):
    global is_speaking
    try:
        is_speaking = True
        filename = "alert_sound.mp3"
        
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
        pygame.mixer.music.unload()
        if os.path.exists(filename):
            os.remove(filename)
    except Exception as e:
        print(f"Audio Error: {e}")
    finally:
        is_speaking = False

cap = cv2.VideoCapture(0)
print("Starting Multi-Object Spatial AI Client...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    height, width, _ = frame.shape
    left_boundary = width // 3
    right_boundary = (width // 3) * 2

    _, img_encoded = cv2.imencode('.jpg', frame)

    try:
        response = requests.post(SERVER_URL, files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")})
        
        very_close_objects = []  

        if response.status_code == 200:
            data = response.json()
            for obj in data.get("detections", []):
                x1, y1, x2, y2 = obj["bbox"]
                cx, cy = obj["center"]
                label = obj["label"]
                status = obj["status"]

                if cx < left_boundary:
                    direction = "on your left"
                elif cx > right_boundary:
                    direction = "on your right"
                else:
                    direction = "ahead"

                if "VERY CLOSE" in status:
                    color = (0, 0, 255)      # Red Box
                    very_close_objects.append(f"{label} {direction}")
                elif "MID" in status:
                    color = (0, 255, 255)    
                else:
                    color = (0, 255, 0)      

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
                cv2.putText(frame, f"{label} | {direction} | {status}", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if len(very_close_objects) > 0:
            current_time = time.time()
            
            if last_speech_time == 0:
                last_speech_time = current_time - GAP_BETWEEN_SPEAKS

            if repeat_count < MAX_REPETITIONS:
                if not is_speaking and (current_time - last_speech_time >= GAP_BETWEEN_SPEAKS):
                    repeat_count += 1
                    last_speech_time = current_time

                    if len(very_close_objects) == 1:
                        alert_text = f"Warning! {very_close_objects[0]}, is very close!"
                    else:
                        objects_sentence = ", and ".join(very_close_objects)
                        alert_text = f"Warning! {objects_sentence}, are very close!"
                    
                    print(f"[ALERT {repeat_count}/{MAX_REPETITIONS}]: {alert_text}")
                    threading.Thread(target=play_audio_file, args=(alert_text,), daemon=True).start()
            else:
                if current_time - last_speech_time >= LONG_PAUSE_COOLDOWN:
                    repeat_count = 0  
                    last_speech_time = current_time - GAP_BETWEEN_SPEAKS

        else:

            if repeat_count < MAX_REPETITIONS:
                repeat_count = 0
                last_speech_time = 0

    except Exception as e:
        cv2.putText(frame, "Server Connecting...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.line(frame, (left_boundary, 0), (left_boundary, height), (255, 255, 255), 1)
    cv2.line(frame, (right_boundary, 0), (right_boundary, height), (255, 255, 255), 1)

    cv2.imshow("Spatial AI - Multi-Object Client", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()