import math
import os
import cv2
import natsort
from deepface import DeepFace

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


class Character:
    def __init__(self, name_char, path_profiles):
        self.name: str = name_char
        self.profile: str = path_profiles
        self.list_appearance: list[float] = []
        self.list_interaction: list[Interaction] = []


class Interaction:
    def __init__(self, timestamp: float, target: str):
        self.timestamp = timestamp
        self.target = target


def identify_character(profiles: list[Character], img_name: str, path_output: str):
    method_distance = [('cosine', 0.4), ('euclidean', 0.6), ('euclidean_l2', 0.86)]
    distance_metric, threshold_max = method_distance[2]
    # 영상에 있는 인물 파악하여 배열로 저장
    result_face_analyze = DeepFace.analyze(img_path=img_name,
                                           actions=('gender', 'emotion'),
                                           detector_backend='retinaface', enforce_detection=False, silent=True)
    # 영상
    img = cv2.imread(img_name)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1
    # 일반적인 감정을 나타내는 글자 색
    color_font_base = (0, 0, 255)
    # 기본 x 여백
    offset_x = 5
    # 증감 y 여백
    margin_y = 10
    # 소수점 반올림 자릿수
    digit = 1
    # 감정을 나타내는 문자 중 가장 길이가 긴 문자의 길이, 문자열 정렬 목적
    len_max_emotion = len(max(list_emotion, key=len))
    # 파일명
    path_array = img_name.split("/")
    dot_array = path_array[-1].split(".")
    origin_file_name = dot_array[0]
    # 영상 내 인물들 순회
    for face_index, face_result in enumerate(result_face_analyze):
        # roi
        point_x, point_y, width, height = face_result['region']['x'], face_result['region']['y'], face_result['region'][
            'w'], face_result['region']['h']
        img_frag = img[point_y:point_y + height, point_x:point_x + width]
        list_distance_min = []
        # 등장인물 순회
        for person in profiles:
            list_image_profile = natsort.natsorted(os.listdir(person.profile))
            # 프로필 순회
            list_distance = []
            for profile in list_image_profile:
                path_verify = f"{person.profile}/{profile}"
                result = DeepFace.verify(img1_path=img_frag, img2_path=path_verify, detector_backend="opencv",
                                         enforce_detection=False, distance_metric=distance_metric)
                print(f"{person.name} - {path_verify} - {result['distance']}")
                list_distance.append(result["distance"])
            list_distance_min.append(min(list_distance))
        min_distance = min(list_distance_min)
        if min_distance > threshold_max:
            name_recognized = None
        else:
            name_recognized = profiles[list_distance_min.index(min(list_distance_min))].name
        if name_recognized is not None:
            name_txt_file = f'{path_output}/emotion.csv'
            if os.path.isfile(name_txt_file):
                f = open(name_txt_file, 'a')
            else:
                f = open(name_txt_file, 'w')
                header = 'name,timestamp'
                for e in list_emotion:
                    header += f",{e}"
                header += "\n"
                f.write(header)
            list_fe = []
            for e in list_emotion:
                list_fe.append(round(face_result['emotion'][e], digit))
            timestamp = origin_file_name.split('_')[1]
            text = f'{name_recognized},{timestamp},{list_fe[0]},{list_fe[1]},{list_fe[2]},' \
                   f'{list_fe[3]},{list_fe[4]},{list_fe[5]},{list_fe[6]}\n'
            f.write(text)
            f.close()
        font_scale = width / 350
        size, base_line = cv2.getTextSize("Sample Text For Test", font_face, font_scale, font_thickness)
        offset_y = size[1]
        cv2.rectangle(img, (point_x, point_y), (point_x + width, point_y + height), (128, 128, 0), 2)
        distance_min_ceil = math.ceil(min_distance * 100) / 100
        if name_recognized is None:
            cv2.putText(img, str(face_index + 1),
                        (point_x + offset_x, point_y + offset_y + margin_y), font_face,
                        font_scale,
                        color_font_base,
                        font_thickness)
        else:
            cv2.putText(img, f'{name_recognized} - {distance_min_ceil}',
                        (point_x + offset_x, point_y + offset_y + margin_y),
                        font_face,
                        font_scale,
                        color_font_base,
                        font_thickness)
        for emotion_index, emotion in enumerate(list_emotion):
            if face_result['dominant_emotion'] == emotion:
                cv2.putText(img, f"{emotion.ljust(len_max_emotion)} {round(face_result['emotion'][emotion], digit)}",
                            (point_x + offset_x, point_y + offset_y * (emotion_index + 2) + margin_y),
                            font_face,
                            font_scale,
                            (0, 255, 0), font_thickness)
            else:
                cv2.putText(img, f"{emotion.ljust(len_max_emotion)} {round(face_result['emotion'][emotion], digit)}",
                            (point_x + offset_x, point_y + offset_y * (emotion_index + 2) + margin_y),
                            font_face,
                            font_scale,
                            color_font_base, font_thickness)
    name_emotion_file = f"{path_output}/{origin_file_name}.png"
    cv2.imwrite(name_emotion_file, img)
    print(f'{name_emotion_file} generated.')
    print('----------------------------------------')


if __name__ == "__main__":
    path_main = './the_man_from_nowhere'
    list_name = ['Cha Tae Sik', 'Jeong So Mi', 'Man Seok', 'Jong Seok', 'Lam Loan']
    list_profile = []
    for name in list_name:
        list_profile.append(Character(name, f'{path_main}/profile/{name.replace(" ", "_").lower()}'))

    path_result = f'{path_main}/shot_emotion'
    if not os.path.exists(path_result):
        os.makedirs(path_result)

    path_indexed = f'{path_main}/indexed_shot'
    list_shot = natsort.natsorted(os.listdir(path_indexed))
    for shot in list_shot:
        identify_character(list_profile, f'{path_indexed}/{shot}', path_result)
