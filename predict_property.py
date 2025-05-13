import os
import re
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image


def natural_key(filename):
    # 숫자와 문자를 나눠서 정렬 키 생성: '10.png' → ['10', '.png']
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', filename)]


# -------------------------
# 설정
# -------------------------
MODEL_FOLDER = 'model'  # .keras 파일들이 들어있는 폴더
IMAGE_FOLDER = 'test'  # 예측할 .png 이미지들이 들어있는 폴더
IMG_SIZE = (224, 224)

# -------------------------
# 모든 모델 파일 불러오기
# -------------------------
model_paths = glob(os.path.join(MODEL_FOLDER, '*.keras'))
model_paths.sort()

if not model_paths:
    raise FileNotFoundError(f"❌ 모델 폴더({MODEL_FOLDER})에 .keras 파일이 없습니다.")

# 속성 이름은 파일명에서 추출 (예: "Smiling.keras" → "Smiling")
attribute_names = [os.path.splitext(os.path.basename(p))[0] for p in model_paths]

# -------------------------
# 모든 이미지 파일 불러오기
# -------------------------
image_paths = glob(os.path.join(IMAGE_FOLDER, '*'))
image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_paths = sorted(image_paths, key=lambda x: natural_key(os.path.basename(x)))

if not image_paths:
    raise FileNotFoundError(f"❌ 이미지 폴더({IMAGE_FOLDER})에 .png 파일이 없습니다.")

# -------------------------
# 예측 수행
# -------------------------
results = []

for img_path in image_paths:
    img_name = os.path.basename(img_path)
    print(img_name)

    # 이미지 로드 및 전처리
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # 하나의 이미지에 대해 모든 모델을 순차적으로 적용
    row_result = {'frame': img_name}

    for model_path, attr in zip(model_paths, attribute_names):
        model = tf.keras.models.load_model(model_path)

        prediction = model.predict(img_array, verbose=0)
        score = float(prediction[0][0])
        predicted_class = int(score > 0.5)

        row_result[f'{attr}'] = round(score, 2)
        print(f'{attr} - {round(score, 2)}')
        # row_result[f'{attr}'] = predicted_class
        # print(f'{attr} - {predicted_class}')

    results.append(row_result)
    print()

# -------------------------
# 결과 저장 및 출력
# -------------------------
df_results = pd.DataFrame(results)
print("\n✅ 예측 결과:")
print(df_results)

# 저장할 경우:
df_results.to_csv('prediction_results.csv', index=False)
print("\n📁 'prediction_results.csv' 파일로 결과 저장 완료!")
