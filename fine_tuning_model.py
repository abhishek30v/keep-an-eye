from ultralytics import YOLO

def main():
    model = YOLO("best.pt")  # Your trained model

    model.train(
        data="animals.yaml",
        epochs=50,
        imgsz=640,
        lr0=0.0001,
        degrees=10,
        scale=0.5,
        shear=2,
        translate=0.1,
        flipud=0.2,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        workers=3,
        batch=16,
        device=0,
        patience=22,
    )

    print("\n✅ Fine-tuning started on best.pt with advanced augmentations.")

# 👇 Required to avoid multiprocessing crash on Windows
if __name__ == "__main__":
    main()
