from idlelib import config

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/train/weights/best.pt')

    results = model.predict(
        source='datasets/overhead_person/test/images',
        save=True,
        conf=0.5
    )