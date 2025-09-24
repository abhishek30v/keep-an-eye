# 📦 Required Libraries
import os
import shutil
from pathlib import Path
from ultralytics import YOLO


# 🧪 Validation Function
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
    val_map = results.box.map
    map50 = results.box.map50
    precision = results.box.mp
    recall = results.box.mr

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


# 🚀 Main Function Entry Point 
if __name__ == "__main__":
    model_path = "best.pt"
    data_yaml_path = "animals.yaml"

    validate_model(model_path, data_yaml_path)