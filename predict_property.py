import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import os
from glob import glob

# -------------------------
# 모델 불러오기
# -------------------------
MODEL_PATH = './No_Beard.keras'  # 저장된 .keras 파일 경로
IMG_HEIGHT, IMG_WIDTH = 224, 224

# 모델 로드
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ 모델이 정상적으로 로드되었습니다.")


# -------------------------
# 폴더 내 이미지들 예측 함수
# -------------------------
def predict_folder(folder_path):
    if not os.path.exists(folder_path):
        print("❌ 입력한 폴더 경로가 존재하지 않습니다.")
        return None

    # 폴더 내 모든 이미지 경로 수집
    image_paths = glob(os.path.join(folder_path, '*'))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(image_paths) == 0:
        print("❌ 폴더 안에 예측할 이미지가 없습니다.")
        return None

    print(f"🔍 {len(image_paths)}장의 이미지를 찾았습니다. 예측을 시작합니다...\n")

    results = {}

    for img_path in image_paths:
        # 이미지 로드 및 전처리
        img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # 예측
        prediction = model.predict(img_array)
        predicted_class = int(prediction[0][0] > 0.5)

        # 결과 저장
        results[os.path.basename(img_path)] = {
            'score': float(prediction[0][0]),
            'predicted_class': predicted_class
        }

        print(
            f"🖼 {os.path.basename(img_path)} → 점수: {prediction[0][0]:.4f}, 예측: {'Positive (1)' if predicted_class else 'Negative (0)'}")

    print("\n✅ 모든 이미지 예측 완료!")
    return results


# -------------------------
# 사용 예시
# -------------------------
# 예측하고 싶은 폴더 경로
test_folder_path = './test'

# 폴더 예측 실행
predict_folder(test_folder_path)
