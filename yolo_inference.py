from ultralytics import YOLO 

model = YOLO('models/train/weights/best.pt')

results = model.predict('input_videos/CV_Task.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)