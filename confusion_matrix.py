import numpy as np

# 오차 행렬
confusion_matrix = np.array([
    [5, 0, 8, 0, 2],
    [0, 19, 1, 3, 0],
    [2, 0, 52, 7, 6],
    [0, 0, 7, 32, 4],
    [2, 0, 4, 2, 10]
])

# 클래스 수
num_classes = confusion_matrix.shape[0]

# 정밀도, 재현율, F1 점수 저장
precision = []
recall = []
f1_scores = []

for i in range(num_classes):
    # True Positive
    tp = confusion_matrix[i, i]

    # False Positive: 해당 열의 합에서 True Positive 제외
    fp = np.sum(confusion_matrix[:, i]) - tp

    # False Negative: 해당 행의 합에서 True Positive 제외
    fn = np.sum(confusion_matrix[i, :]) - tp

    # 정밀도와 재현율 계산
    precision_i = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_i = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 점수 계산
    f1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i) if (precision_i + recall_i) > 0 else 0

    # 저장
    precision.append(precision_i)
    recall.append(recall_i)
    f1_scores.append(f1_i)

# 결과 출력
for i in range(num_classes):
    print(f"Class {i}: Precision={precision[i]:.2f}, Recall={recall[i]:.2f}, F1-Score={f1_scores[i]:.2f}")

macro_f1 = np.mean(f1_scores)
print(f"Macro F1-Score: {macro_f1:.2f}")

class_support = np.sum(confusion_matrix, axis=1)
weighted_f1 = np.sum(np.array(f1_scores) * class_support) / np.sum(class_support)
print(f"Weighted F1-Score: {weighted_f1:.2f}")
