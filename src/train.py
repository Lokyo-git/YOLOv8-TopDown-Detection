from onnx.gen_proto import translate
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='config/data.yaml',
        epochs=80,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,

        degrees=10.0,
        scale=0.2,
        translate=0.1,
        fliplr=0.5,
        flipud=0.0,
        perspective=0.0005,

        hsv_h=0.015,
        hsv_v=0.7,
        hsv_s=0.4,

        mosaic=0.8,
        mixup=0.15
    )