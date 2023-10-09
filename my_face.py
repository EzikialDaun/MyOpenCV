import cv2
from deepface import DeepFace

path_data = './data/'


def cmp_detect(img_name):
    # 공통점 - 나이: 하, 인종: 중상
    # opencv - 성별: 중, 인식 민감도: 하, 감정: 하, 속도: 빠름
    # ssd - 성별: 상, 인식 민감도: 하, 감정: 중, 속도: 빠름
    # mtcnn - 성별: 중, 인식 민감도: 상, 감정: 중상: 속도: 느림
    # retinaface - 성별: 하, 인식 민감도: 중상, 감정: 중, 속도: 느림
    backends = [
        'opencv',
        'ssd',
        'mtcnn',
        'retinaface',
    ]

    for backend in backends:
        try:
            objs = DeepFace.analyze(img_path=img_name,
                                    actions=('age', 'gender', 'race', 'emotion'),
                                    detector_backend=backend,
                                    )
            img = cv2.imread(img_name)

            for i, obj in enumerate(objs):
                key_region = 'region'

                point_x, point_y, width, height = obj[key_region]['x'], obj[key_region]['y'], obj[key_region]['w'], \
                                                  obj[key_region]['h']

                cv2.rectangle(img, (point_x, point_y), (point_x + width, point_y + height), (0, 0, 255), 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(i + 1), (point_x, point_y), font, 0.7, (0, 0, 255), 1)

                print(f"detector - {backend}")
                print(f"index - {str(i + 1)}")
                print(f"region - {obj[key_region]}")
                print(f"age - {obj['age']}")
                print(f"gender - {obj['dominant_gender']} - {obj['gender']}")
                print(f"race - {obj['dominant_race']} - {obj['race']}")
                print(f"emotion - {obj['dominant_emotion']} - {obj['emotion']}")
                print()

            cv2.imshow('image - ' + backend, img)
        except Exception as e:
            print(f"detector - {backend}")
            print(e)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cmp_detect(path_data + 'hq720_2.jpg')
