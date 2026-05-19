from idlelib import config

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/train4/weights/best.pt')

    results = model.predict(
        source='/home/hnucv/yolo_person_overhead/dataset/test/new_images',
        save=True,
        conf=0.5
    )