import torch
import cv2
from numpy import random
import sys

# YOLOv9 모델 정의
model_path = 'model/best.pt'
yolov9_path = 'yolov9'
sys.path.insert(0, yolov9_path)
model = torch.hub.load(yolov9_path, 'custom', path=model_path, source='local', force_reload=True)

if torch.cuda.is_available():
    model = model.cuda()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # 프레임 크기 조정
        frame = cv2.resize(frame, (640, 480))

        # 이미지를 모델에 입력
        results = model(frame)

        # 객체 감지 결과 얻기
        detections = results.pandas().xyxy[0]

        if not detections.empty:
            # 결과를 반복하며 객체 표시
            for _, detection in detections.iterrows():
                x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']].astype(int).values
                label = detection['name']
                conf = detection['confidence']

                # 박스와 라벨 표시
                if conf > 0.4:
                    color = [int(c) for c in random.choice(range(256), size=3)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 프레임 표시
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
