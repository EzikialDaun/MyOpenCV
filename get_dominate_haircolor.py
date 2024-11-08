from deepface import DeepFace
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import cv2
import numpy as np
from sklearn.cluster import KMeans


def detect_hat(img_hat):
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="Ug46qshslegLKtEbk8X8"
    )

    custom_configuration = InferenceConfiguration(confidence_threshold=0.7)
    with CLIENT.use_configuration(custom_configuration):
        result_hat = CLIENT.infer(img_hat, model_id="yolov8-hat-detection/3")
        if len(result_hat["predictions"]) > 0:
            return "1"
        else:
            return None


def has_black_hair(colors):
    for c in colors:
        if c[1] <= 120 and c[2] <= 20:
            return "1"
    return "0"


# 이미지 내 머리 영역 추출 함수
def extract_hair_area(img, limit_cluster=3):
    CLIENT = InferenceHTTPClient(
        api_url="https://outline.roboflow.com",
        api_key="KmRNa5J58vhavwNUSa7b"
    )
    # result = CLIENT.infer(img, model_id="hairseg-zr4z4/4")
    custom_configuration = InferenceConfiguration(confidence_threshold=0.7)
    with CLIENT.use_configuration(custom_configuration):
        result = CLIENT.infer(img, model_id="hairseg-zr4z4/4")
        result_sorted = sorted(result['predictions'], key=lambda x: x["confidence"], reverse=True)
    points = result_sorted[0]['points']
    temp = []
    for point in points:
        temp.append([point['x'], point['y']])

    p = np.array([temp], dtype=np.int32)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [p], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(img, img, mask=mask)
    bg = np.ones_like(img, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst = bg + dst

    frame = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    frame = frame.reshape((frame.shape[0] * frame.shape[1], 3))
    kmeans = KMeans(n_clusters=limit_cluster)
    kmeans.fit(frame)
    centers = kmeans.cluster_centers_.astype(int)

    return centers, dst


if __name__ == '__main__':
    path_src = '..\\..\\lab\\MyFace Dataset Lite\\django_unchained\\probe\\468.png'
    analyzed = DeepFace.analyze(
        img_path=path_src,
        actions=['gender', 'race'],
        detector_backend="retinaface",
        enforce_detection=False,
    )

    if len(analyzed) > 0:
        src = cv2.imread(path_src)
        x, y, w, h = (analyzed[0]["region"][key] for key in ("x", "y", "w", "h"))
        padding = int((w + h) / 4)

        y_start = max(0, y - padding)
        y_end = min(src.shape[0], y + int((h + padding) / 2))
        # y_end = min(src.shape[1], y + h + padding)
        x_start = max(0, x - padding)
        x_end = min(src.shape[1], x + w + padding)

        frag = src[y_start:y_end, x_start:x_end]

        dominant_hsv, masked = extract_hair_area(frag)
        print(dominant_hsv)

        # 빈 캔버스 생성
        size = 100
        height = size
        width = size * len(dominant_hsv)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # HSV 색상을 BGR로 변환하여 캔버스에 채우기
        for i, hsv in enumerate(dominant_hsv):
            # HSV 배열을 NumPy 배열로 변환 후 BGR로 변경
            bgr_color = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0]
            # 캔버스의 각 색상 영역 채우기
            canvas[:, i * size:(i + 1) * size] = bgr_color

        # 결과 이미지 출력
        cv2.imshow("src", src)
        cv2.imshow("frag", frag)
        cv2.imshow("masked", masked)
        cv2.imshow("HSV Colors", canvas)
        cv2.waitKey()
        cv2.destroyAllWindows()
