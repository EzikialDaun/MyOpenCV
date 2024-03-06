import copy
import glob
import math
import os
import cv2
import natsort
from deepface import DeepFace


def identify_character(profiles, img_name, path_output, threshold_dist=0.86):
    method_distance = [('cosine', 0.4), ('euclidean', 0.6), ('euclidean_l2', 0.86)]
    distance_metric, threshold_max = method_distance[2]
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
    # 영상에 있는 인물 파악하여 배열로 저장
    face_analyze_result = DeepFace.analyze(img_path=img_name,
                                           actions=('gender', 'emotion'),
                                           detector_backend='retinaface', enforce_detection=False, silent=True)
    # 영상
    img = cv2.imread(img_name)
    # 폰트
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    # 글씨 두께
    font_thickness = 1
    # 일반적인 감정을 나타내는 글자 색
    color_font_base = (0, 0, 255)
    # 지배적인 감정을 나타내는 글자 색
    color_font_dominant = (0, 255, 0)
    # 기본 x 여백
    offset_x = 5
    # 증감 y 여백
    margin_y = 10
    # 소수점 반올림 자릿수
    digit = 1
    # 감정을 나타내는 문자 중 가장 길이가 긴 문자의 길이, 문자열 정렬 목적
    len_just = len('surprise')
    copied_profiles = copy.deepcopy(profiles)
    # 폰트 크기 상수
    ratio_font = 350
    # 파일명
    path_array = img_name.split("/")
    dot_array = path_array[-1].split(".")
    origin_file_name = dot_array[0]
    # 영상 내 인물의 수 만큼 순회
    for face_index, face_result in enumerate(face_analyze_result):
        if len(copied_profiles) == 0:
            break
        # roi
        point_x, point_y, width, height = face_result['region']['x'], face_result['region']['y'], face_result['region'][
            'w'], face_result['region']['h']
        img_frag = img[point_y:point_y + height, point_x:point_x + width]
        # 등장인물 프로필에 등록된 인물일 경우, 중복 체크를 제거하기 위한 해당 프로필의 인덱스
        list_distance_min = []
        for profile in copied_profiles:
            dfs = DeepFace.find(img_path=img_frag, db_path=profile.profile, enforce_detection=False,
                                detector_backend='opencv', distance_metric=distance_metric, model_name='VGG-Face',
                                silent=True)
            len_dfs = dfs[0].shape[0]
            if len_dfs == 0:
                list_distance_min.append(threshold_max)
                print(f'{profile.name} not found\n')
                continue
            sum_result = 0
            for result_find in dfs[0].values:
                sum_result += result_find[5]
                print(f'{result_find[0]} - {result_find[5]}')
            len_profile = 0
            for i in ['*.png', '*.jpg', '*.jpeg']:
                len_profile += len(glob.glob(f'{profile.profile}/{i}'))
            len_add = len_profile - len_dfs
            for _ in range(len_add):
                sum_result += threshold_max
                print(f'{profile.name} - penalty - {threshold_max}')
            avg_result = sum_result / len_profile
            list_distance_min.append(avg_result)
            print(f'{profile.name} - avg - {avg_result}\n')
        distance_min = min(list_distance_min)
        index_correct = list_distance_min.index(min(list_distance_min))
        name_similar = copied_profiles[index_correct].name
        copied_profiles.pop(index_correct)
        # 인수로 설정된 값보다 유사도가 낮으면 무시
        if distance_min > threshold_dist:
            name_similar = None
        if name_similar is not None:
            name_txt_file = f'{path_output}/emotion.csv'
            f = None
            if os.path.isfile(name_txt_file):
                f = open(name_txt_file, 'a')
            else:
                f = open(name_txt_file, 'w')
            list_fe = []
            for e in list_emotion:
                list_fe.append(round(face_result['emotion'][e], digit))
            timestamp = origin_file_name.split('_')[1]
            text = f'{name_similar},{timestamp},{list_fe[0]},{list_fe[1]},{list_fe[2]},{list_fe[3]},{list_fe[4]},{list_fe[5]},{list_fe[6]}\n '
            f.write(text)
            f.close()

        font_scale = width / ratio_font
        size, base_line = cv2.getTextSize("Sample Text For Test", font_face, font_scale, font_thickness)
        offset_y = size[1]
        cv2.rectangle(img, (point_x, point_y), (point_x + width, point_y + height), (128, 128, 0), 2)
        distance_min_ceil = math.ceil(distance_min * 100) / 100
        if name_similar is None:
            cv2.putText(img, str(face_index + 1),
                        (point_x + offset_x, point_y + offset_y + margin_y), font_face,
                        font_scale,
                        color_font_base,
                        font_thickness)
        else:
            cv2.putText(img, f'{name_similar} - {distance_min_ceil}',
                        (point_x + offset_x, point_y + offset_y + margin_y),
                        font_face,
                        font_scale,
                        color_font_base,
                        font_thickness)
        for emotion_index, emotion in enumerate(list_emotion):
            if face_result['dominant_emotion'] == emotion:
                cv2.putText(img, f"{emotion.ljust(len_just)} {round(face_result['emotion'][emotion], digit)}",
                            (point_x + offset_x, point_y + offset_y * (emotion_index + 2) + margin_y),
                            font_face,
                            font_scale,
                            color_font_dominant, font_thickness)
            else:
                cv2.putText(img, f"{emotion.ljust(len_just)} {round(face_result['emotion'][emotion], digit)}",
                            (point_x + offset_x, point_y + offset_y * (emotion_index + 2) + margin_y),
                            font_face,
                            font_scale,
                            color_font_base, font_thickness)
    name_emotion_file = f"{path_output}/{origin_file_name}.png"
    cv2.imwrite(name_emotion_file, img)
    print(f'{name_emotion_file} generated.')
    print('----------------------------------------')


class Character:
    def __init__(self, name_char, path_profiles):
        self.name = name_char
        self.profile = path_profiles
        self.list_emotion = []


class Emotion:
    def __init__(self, array_emotion, timestamp):
        self.array_emotion = array_emotion
        self.timestamp = timestamp


if __name__ == "__main__":
    path_main = './the_man_from_nowhere'
    list_name = ['Cha Tae Sik 1', 'Cha Tae Sik 2', 'Jeong So Mi', 'Man Seok', 'Jong Seok', 'Lam Loan']
    list_profile = []
    for name in list_name:
        list_profile.append(Character(name, f'{path_main}/profile/{name.replace(" ", "_").lower()}'))

    path_result = f'{path_main}/shot_emotion'
    if not os.path.exists(path_result):
        os.makedirs(path_result)

    path_indexed = f'{path_main}/indexed_shot'
    list_shot = natsort.natsorted(os.listdir(path_indexed))
    for shot in list_shot:
        identify_character(list_profile, f'{path_indexed}/{shot}',
                           path_result, threshold_dist=0.8)
