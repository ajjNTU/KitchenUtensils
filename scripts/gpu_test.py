import torch
import ultralytics

def test_gpu_setup():
    print("=== PyTorch GPU Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    print("\n=== YOLOv8 GPU Test ===")
    print(f"Ultralytics version: {ultralytics.__version__}")
    
    # Test a simple YOLOv8 operation
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # Load smallest model for testing
        print("Successfully loaded YOLOv8 model")
        print("GPU is ready for YOLOv8 training!")
    except Exception as e:
        print(f"Error testing YOLOv8: {str(e)}")

if __name__ == "__main__":
    test_gpu_setup()