import cv2
import mediapipe as mp
import numpy as np
import os

print("Initializing Camera and MediaPipe...")

try:
    # Force Python to reveal the hidden internal error
    import mediapipe.python.solutions as mp_solutions
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
except Exception as e:
    print("\n" + "="*60)
    print(f"HIDDEN MEDIAPIPE ERROR REVEALED: {e}")
    print("="*60)
    print("Fix 1: Run `pip install tensorflow==2.15.0 mediapipe==0.10.9 protobuf==3.20.3 jax==0.4.23 jaxlib==0.4.23` in your terminal.")
    print("Fix 2: If the error says 'No module named', you are using an unsupported Python version (like 3.12). You must uninstall it and install Python 3.11.")
    print("="*60 + "\n")
    raise SystemExit("Exiting due to corrupted MediaPipe environment.")

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

SEQUENCE_LENGTH = 10
sequence = []
X_data = []
y_data = []

print("Connecting to webcam...")
cap = None

# Auto-scan camera indices 0 to 3 to bypass virtual/broken cameras
for cam_idx in range(4):
    print(f" -> Testing camera index {cam_idx}...")
    # Test DSHOW first on Windows as it is much more stable than the default MSMF
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == 'nt' else [cv2.CAP_ANY]
    
    for backend in backends:
        temp_cap = cv2.VideoCapture(cam_idx, backend)
        if temp_cap.isOpened():
            is_valid = False
            # Read up to 20 frames to allow hardware to warm up and verify real pixel data
            for _ in range(20):
                ret, frame = temp_cap.read()
                # Check if frame has actual visible data (not a pure black placeholder)
                if ret and frame is not None and np.max(frame) > 0:
                    is_valid = True
                    break
            if is_valid:
                cap = temp_cap
                break
            temp_cap.release()
    
    if cap is not None:
        # Force hardware to a fast, low resolution to prevent severe lag
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f" -> SUCCESS: Connected to physical camera at index {cam_idx}!")
        break

if cap is None:
    print("\nERROR: All cameras returned a black screen or failed to open.")
    print("1. Ensure your laptop's physical privacy shutter is open.")
    print("2. Make sure app.py is completely stopped (it might be holding the camera).")
    raise SystemExit("Exiting...")

frame_count = 0

print("\n--- CONTROLS ---")
print("Act out a motion, then press:")
print(" '0' -> Save last 10 frames as NORMAL")
print(" '1' -> Save last 10 frames as BRAWL")
print(" '2' -> Save last 10 frames as FALL")
print(" 'q' -> Save and Quit")
print("----------------\n")

pose_detected = False
empty_frames = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        empty_frames += 1
        if empty_frames > 20:
            print("\nERROR: Camera signal lost!")
            break
        continue
    
    empty_frames = 0
    frame_count += 1
    
    # Ensure standard processing size
    frame = cv2.resize(frame, (640, 480))
    # We only process every 5th frame to perfectly match the speed of app.py (~1.6s of action)
    run_ai = (frame_count % 5 == 0)
    
    if run_ai:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            pose_detected = True
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            
            # Center extraction based on Hips (Same as app.py)
            cx = (landmarks[23].x + landmarks[24].x) / 2
            cy = (landmarks[23].y + landmarks[24].y) / 2
            cz = (landmarks[23].z + landmarks[24].z) / 2
            
            current_pose = []
            for res in landmarks:
                current_pose.extend([res.x - cx, res.y - cy, res.z - cz, res.visibility])
                
            sequence.append(current_pose)
            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)
        else:
            pose_detected = False
                
    # Overlay UI text
    color = (0, 255, 0) if len(sequence) == SEQUENCE_LENGTH else (0, 0, 255)
    cv2.putText(frame, f"AI Buffer: {len(sequence)}/{SEQUENCE_LENGTH}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if not pose_detected:
        cv2.putText(frame, "NO POSE DETECTED! Step back.", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
    cv2.imshow('Data Collection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('0'), ord('1'), ord('2')]:
        if len(sequence) == SEQUENCE_LENGTH:
            label = int(chr(key))
            X_data.append(sequence.copy())
            y_data.append(label)
            print(f" -> Saved sequence for Class {label}. Total recorded: {len(X_data)}")
        else:
            print(f" -> Buffer not full yet ({len(sequence)}/{SEQUENCE_LENGTH}). Step back so the camera can see your body!")

cap.release()
cv2.destroyAllWindows()

if X_data:
    print(f"\nSaving {len(X_data)} new samples to disk...")
    # Append to existing dataset files if they exist, otherwise create them
    X_save = np.concatenate((np.load('X_data.npy'), X_data)) if os.path.exists('X_data.npy') else np.array(X_data)
    y_save = np.concatenate((np.load('y_data.npy'), y_data)) if os.path.exists('y_data.npy') else np.array(y_data)
    np.save('X_data.npy', X_save)
    np.save('y_data.npy', y_save)
    print("Success! You can now run train_lstm_model.py")