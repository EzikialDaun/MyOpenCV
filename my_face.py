import copy

import cv2
from deepface import DeepFace

path_data = './data/'


def identify_character(profiles, img_name):
    face_analyze_result = DeepFace.analyze(img_path=img_name,
                                           actions=('age', 'gender', 'race', 'emotion'),
                                           detector_backend='retinaface')
    img = cv2.imread(img_name)

    # 전처리
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    # 글씨 두께
    font_thickness = 1
    # 일반적인 감정을 나타내는 글자 색
    font_base_color = (0, 0, 255)
    # 지배적인 감정을 나타내는 글자 색
    font_dominant_color = (0, 255, 0)
    # 감정 목록
    list_emotion = [
        'angry',
        'disgust',
        'fear',
        'happy',
        'sad',
        'surprise',
        'neutral'
    ]
    # 기본 x 여백
    offset_x = 5
    # 증감 y 여백
    margin_y = 10
    # 소수점 반올림 자릿수
    digit = 1
    # 감정을 나타내는 문자 중 가장 길이가 긴 문자의 길이, 문자열 정렬 목적
    len_just = len('surprise')
    copied_profiles = copy.deepcopy(profiles)
    # 영상 내 인물의 수 만큼 순회
    for face_index, face_result in enumerate(face_analyze_result):
        # roi
        point_x, point_y, width, height = face_result['region']['x'], face_result['region']['y'], face_result['region'][
            'w'], face_result['region']['h']

        # roi로 추출한 인물의 얼굴 영역
        img_frag = img[point_y:point_y + height, point_x:point_x + width]
        # 등장인물 사전에 등록된 인물일 경우, 그 인물의 이름을 저장하는 용도
        char_name = None
        # 등장인물 사전에 등록된 인물일 경우, 중복 체크를 제거하기 위한 해당 사전의 인덱스
        delete_index = None
        for p_index, profile in enumerate(copied_profiles):
            verify_result = DeepFace.verify(img1_path=profile.profile, img2_path=img_frag,
                                            enforce_detection=False,
                                            detector_backend='retinaface', model_name='Facenet',
                                            distance_metric='cosine')
            print(verify_result)
            # 등장인물과 프로필의 유사도 판단 기준
            if verify_result['distance'] <= 0.46:
                char_name = profile.name
                delete_index = p_index
                break
        if delete_index is not None:
            copied_profiles.pop(delete_index)

        font_scale = width / 350
        size, base_line = cv2.getTextSize("Sample Text For Test", font_face, font_scale, font_thickness)
        offset_y = size[1]

        cv2.rectangle(img, (point_x, point_y), (point_x + width, point_y + height), (128, 128, 0), 2)
        if char_name is None:
            cv2.putText(img, str(face_index + 1), (point_x + offset_x, point_y + offset_y + margin_y), font_face,
                        font_scale,
                        font_base_color,
                        font_thickness)
        else:
            cv2.putText(img, char_name, (point_x + offset_x, point_y + offset_y + margin_y), font_face,
                        font_scale,
                        font_base_color,
                        font_thickness)

        for emotion_index, emotion in enumerate(list_emotion):
            if face_result['dominant_emotion'] == emotion:
                cv2.putText(img, f"{emotion.ljust(len_just)} {round(face_result['emotion'][emotion], digit)}",
                            (point_x + offset_x, point_y + offset_y * (emotion_index + 2) + margin_y),
                            font_face,
                            font_scale,
                            font_dominant_color, font_thickness)
            else:
                cv2.putText(img, f"{emotion.ljust(len_just)} {round(face_result['emotion'][emotion], digit)}",
                            (point_x + offset_x, point_y + offset_y * (emotion_index + 2) + margin_y),
                            font_face,
                            font_scale,
                            font_base_color, font_thickness)

    # 후처리
    # cv2.imshow(backend, img)
    path_array = img_name.split("/")
    dot_array = path_array[-1].split(".")
    origin_file_name = dot_array[0]
    cv2.imwrite(f"./emotion_shots/{origin_file_name}_retina.png", img)
    # 프로그램 종료 방지, 아무 키 누르면 프로그램 종료
    # cv2.waitKey()
    # 모든 창 소멸
    # cv2.destroyAllWindows()


class Character:
    def __init__(self, name, profile):
        self.name = name
        self.profile = profile


if __name__ == "__main__":
    profile_list = [
        Character('William Thacker', path_data + 'hugh_grant.png'),
        Character('Anna Scott', path_data + 'julia_roberts.png')
    ]
    image_list = [
        path_data + '001.png',
        path_data + '002.png',
        path_data + '003.png',
        path_data + '004.png',
        path_data + '005.png',
        path_data + '006.png',
    ]
    for image in image_list:
        identify_character(profile_list, image)
