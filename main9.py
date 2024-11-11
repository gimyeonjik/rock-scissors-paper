import torch
import cv2
from numpy import random
from ultralytics import YOLO

# YOLO 모델 정의
model = YOLO('/Users/jun/Documents/rsp/best9m.pt')

if torch.cuda.is_available():
    model = model.cuda()

# 가위바위보 승패 판정 함수
def judge(player_sign, opponent_sign):
    # 승리 규칙 정의
    rules = {'Rock': 'Scissors', 'Scissors': 'Paper', 'Paper': 'Rock'}
    if player_sign == opponent_sign:
        return 0  # 무승부
    elif rules[player_sign] == opponent_sign:
        return 1  # 승리
    else:
        return -1  # 패배

# 플레이어에게 추천 패를 계산하는 함수
def recommend_best_sign(player_signs, opponent_signs):
    max_score = float('-inf')
    best_sign = None

    for player_sign in player_signs:
        total_score = 0
        for opponent_sign in opponent_signs:
            score = judge(player_sign, opponent_sign)
            total_score += score
        if total_score > max_score:
            max_score = total_score
            best_sign = player_sign

    return best_sign, max_score

cap = cv2.VideoCapture(0)

conf_threshold = 0.5

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # 프레임 크기 조정
        frame = cv2.resize(frame, (1280, 720))

        # 이미지를 모델에 입력
        results = model(frame, conf=conf_threshold, iou=0.5)

        # 객체 감지 결과 얻기
        detections = results[0].boxes

        # 화면 중앙에 가로선 그리기
        height, width, _ = frame.shape
        mid_y = height // 2
        cv2.line(frame, (0, mid_y), (width, mid_y), (0, 255, 0), 2)

        # 플레이어와 상대방의 패를 저장할 리스트 초기화
        player_signs = []
        opponent_signs = []

        if detections is not None and detections.shape[0] > 0:
            # 결과를 반복하며 객체 표시
            for detection in detections:
                conf = detection.conf[0].item()

                # 신뢰도 기준 적용
                if conf >= conf_threshold:
                    # 좌표와 클래스 라벨 가져오기
                    x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
                    cls_id = int(detection.cls[0].item())
                    label = results[0].names[cls_id]

                    # 박스와 라벨 표시
                    color = [int(c) for c in random.choice(range(256), size=3)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # 객체의 중심 좌표 계산
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # 중복된 라벨 방지
                    if label not in player_signs + opponent_signs:
                        # 객체가 화면의 위쪽 절반에 있는지 아래쪽 절반에 있는지 확인
                        if cy < mid_y:
                            # 상대방의 패 저장 (최대 2개)
                            if len(opponent_signs) < 2:
                                opponent_signs.append(label)
                        else:
                            # 플레이어의 패 저장 (최대 2개)
                            if len(player_signs) < 2:
                                player_signs.append(label)

        # 상대방의 패에 기반하여 플레이어에게 추천 패 계산
        if opponent_signs and player_signs:
            recommended_sign, max_score = recommend_best_sign(player_signs, opponent_signs)
            recommendation_text = f"Recommendation: {recommended_sign} (Score: {max_score})"
        else:
            recommendation_text = 'Waiting for signs...'

        # 플레이어와 상대방의 패를 화면에 표시
        cv2.putText(frame, f'Player: {player_signs}', (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f'Opponent: {opponent_signs}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # 추천 패를 화면에 표시
        cv2.putText(frame, recommendation_text, (10, height // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 프레임 표시
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
