import os
from ultralytics import YOLO
import torch
from pathlib import Path
import json
import shutil

def setup_training_environment():
    """Setup training environment and check GPU availability"""
    print("🔧 Setting up training environment...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        cpu_or_gpu = 'cuda'
    else:
        print("⚠️  GPU not available, using CPU (training will be slower)")
        cpu_or_gpu = 'cpu'
    
    return cpu_or_gpu

def train_yolo_model(data_yaml_path, epochs, batch_size, img_size, patience, project_name, experiment_name, device):

    print("🚀 Starting YOLOv8 Training...")
    
    # Load model
    model = YOLO('last.pt')
    
    # Training parameters
    training_args = {
        # Core training parameters
        'data': data_yaml_path,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'patience': patience,
        'project': project_name,
        'name': experiment_name,
        'workers': 0,
        'save': True,
        'save_period': 15,  # Save checkpoint every 10 epochs
        'cache': False,      # Cache images for faster training
        'device': device,   # cpu or gpu will be used for training the model
        'resume': True,
        
        # Learning
        'lr0': 0.008,           # initial
        'lrf': 0.001,            # final
        'weight_decay': 0.0004,
        'optimizer': 'AdamW',
        'cos_lr': True,

        # Loss function tuning
        'box': 5.0,
        'cls': 0.8,
        'dfl': 1.0,

        # Regularization  
        'label_smoothing': 0.10,
        'erasing': 0.08,
        'fraction': 0.85,
        'dropout': 0.15,

        # Augmentations
        'hsv_h': 0.015,
        'hsv_s': 0.6,
        'hsv_v': 0.3,
        'degrees': 5.0,
        'translate': 0.08,
        'scale': 0.5,
        'shear': 1.0,
        'flipud': 0.0,
        'fliplr': 0.4,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'mosaic': 0.7,
        
        'amp': False,
        'single_cls': False,
        
        # Validation settings
        'val': True,
        'plots': True,      # Generate training plots
        'verbose': True,    # Verbose output
    }
    
    # Start training
    print(f"📊 Training Configuration:")
    print(f"   Model: YOLOv8n")
    print(f"   Epochs: {epochs} (with early stopping patience: {patience})")
    print(f"   Batch Size: {batch_size}")
    print(f"   Image Size: {img_size}")
    print(f"   Data: {data_yaml_path}")
    
    try:
        results = model.train(**training_args)
        print("✅ Training completed successfully!")
        
        # Print best results
        print(f"\n📈 Best Results:")
        print(f"   Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"   Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return None

def validate_model(model_path, data_yaml_path):
    """Validate the trained model"""
    print("🔍 Validating trained model...")
    
    model_path = Path(model_path)
    model = YOLO(model_path)
    results = model.val(
        data=data_yaml_path,
        split='val',
        conf=0.25,
        iou=0.6,
        plots=True,
        save_json=False,
        )
    
    # Extract validation metrics directly
    metrics = results.metrics
    val_map = metrics.map50to95
    map50 = metrics.map50
    precision = metrics.precision
    recall = metrics.recall

    # Print comprehensive report
    print("\n📊 Validation Report:")
    print(f"mAP50-95: {val_map:.4f}")
    print(f"mAP50: {map50:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Generate visual report
    print("\n📈 Generating visual report...")
    report_dir = model_path.parent / "validation_report"
    os.makedirs(report_dir, exist_ok=True)

    # Copy visualizations
    for viz_file in ['confusion_matrix.png', 'results.png', 'labels.jpg']:
        src = model_path.parent / viz_file
        if src.exists():
            shutil.copy(src, report_dir / viz_file)

    print(f"✅ Validation completed! Report saved to {report_dir}")
    return results

def main():
    """Main training pipeline using existing animals.yaml"""
    
    # 🔧 CONFIGURATION - USING animals.yaml
    ANIMALS_YAML_PATH = "animals.yaml"  # Path to animals.yaml file
    
    # Training configuration
    EPOCHS = 150          # Will use early stopping
    BATCH_SIZE = 24       # Adjust based on your GPU memory
    PATIENCE = 25        # Early stopping patience
    IMG_SIZE = 640
    
    print("🎯 YOLOv8 Custom Training Pipeline")
    print("📂 Using animals.yaml configuration")
    print("🐅 Classes: Tiger, Leopard, Cheetah, Elephant, Monkey, Deer, Lion, Bear, Pig, Bull")
    print("=" * 70)
    
    # Step 1: Setup environment
    cpu_or_gpu = setup_training_environment()
    
    # Step 2: Verify animals.yaml exists
    if not os.path.exists(ANIMALS_YAML_PATH):
        print(f"❌ Error: {ANIMALS_YAML_PATH} not found!")
        print("📋 Please ensure animals.yaml is in the same directory as this script")
        return
    
    print(f"✅ Found {ANIMALS_YAML_PATH}")
    
    # Step 3: Train model using your animals.yaml
    results = train_yolo_model(
        data_yaml_path=ANIMALS_YAML_PATH, 
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        patience=PATIENCE,
        project_name='ProjectX',
        experiment_name='yolov8n_animals',
        device=cpu_or_gpu
    )
    
    if results:
        # Step 4: Get best model path
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        print(f"\n🎉 Training Complete!")
        print(f"📁 Best model saved at: {best_model_path}")
        print(f"📁 Training results saved at: {results.save_dir}")
        
        # Optional: Validate the best model
        print("\n" + "="*70)
        validate_model(str(best_model_path), ANIMALS_YAML_PATH)
        
        print(f"\n🚀 Ready for inference! Use: YOLO('{best_model_path}')")
    
    else:
        print("❌ Training failed. Please check your dataset structure and paths.")

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        os.system("pip install ultralytics")
        from ultralytics import YOLO
    
    main()