from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO("yolo11s.pt")
    # 开始训练
    results = model.train(
        data="custom_data.yaml",
        epochs=25,
        imgsz=96,
        batch=64,
        device='0',
        project="run"
    )