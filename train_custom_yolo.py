from ultralytics import YOLO
import os

if __name__ == '__main__':
    print("Building Custom YOLO Weapon Detector...")
    
    # Upgrade to the 'Small' model instead of 'Nano' for much higher accuracy on small objects
    model = YOLO('yolov8s.pt')
    
    # MAKE SURE the path below matches the extracted dataset folder in your directory!
    # For example, if you downloaded a folder named 'weapon-dataset-1', change it below:
    dataset_yaml_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'Sharp object detection.v6i.yolov8', 'data.yaml')
    
    if not os.path.exists(dataset_yaml_path):
        print(f"ERROR: Could not find {dataset_yaml_path}")
        print("Please download a dataset from Roboflow and extract it into your folder first.")
        exit()

    print("Starting Training (This may take a while depending on your CPU/GPU)...")
    
    # Train the model
    model.train(data=dataset_yaml_path, epochs=100, imgsz=640, batch=16, name='custom_weapon_model')
    
    print("\nSUCCESS! Check the 'runs/detect/custom_weapon_model/weights/' folder for your 'best.pt' file.")