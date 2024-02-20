import copy
import os
import cv2
import natsort
from deepface import DeepFace

# 데이터 디렉토리 경로
PATH_DATA = './data/'

GENDER_MALE = 0
GENDER_FEMALE = 1

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


def identify_character(profiles, img_name, path_output, threshold_dist=0.86):
    # 영상에 있는 인물 파악하여 배열로 저장
    face_analyze_result = DeepFace.analyze(img_path=img_name,
                                           actions=('gender', 'emotion'),
                                           detector_backend='retinaface', enforce_detection=False)
    # 영상
    img = cv2.imread(img_name)
    # 전처리
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    # 폰트
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    # 글씨 두께
    font_thickness = 1
    # 일반적인 감정을 나타내는 글자 색
    font_base_color = (0, 0, 255)
    # 지배적인 감정을 나타내는 글자 색
    font_dominant_color = (0, 255, 0)
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
        # roi
        point_x, point_y, width, height = face_result['region']['x'], face_result['region']['y'], face_result['region'][
            'w'], face_result['region']['h']
        # roi로 추출한 인물의 얼굴 영역
        img_frag = img[point_y:point_y + height, point_x:point_x + width]
        # 등장인물 사전에 등록된 인물일 경우, 그 인물의 이름을 저장하는 용도
        char_name = None
        # 등장인물 사전에 등록된 인물일 경우, 중복 체크를 제거하기 위한 해당 사전의 인덱스
        list_candidate = []
        # 등장인물 후보 탐색
        for p_index, profile in enumerate(copied_profiles):
            verify_result = DeepFace.verify(img1_path=profile.profile, img2_path=img_frag,
                                            enforce_detection=False,
                                            detector_backend='retinaface', model_name='VGG-Face',
                                            distance_metric='euclidean_l2')
            print(verify_result)
            list_candidate.append(verify_result['distance'])
        # 등장인물에 해당되면
        if min(list_candidate) <= threshold_dist:
            index_correct = list_candidate.index(min(list_candidate))
            char_name = copied_profiles[index_correct].name
            copied_profiles.pop(index_correct)
            name_txt_file = f'{path_output}/emotion.txt'
            f = None
            if os.path.isfile(name_txt_file):
                f = open(name_txt_file, 'a')
            else:
                f = open(name_txt_file, 'w')
            f_angry = round(face_result['emotion']['angry'], digit)
            f_disgust = round(face_result['emotion']['disgust'], digit)
            f_fear = round(face_result['emotion']['fear'], digit)
            f_happy = round(face_result['emotion']['happy'], digit)
            f_sad = round(face_result['emotion']['sad'], digit)
            f_surprise = round(face_result['emotion']['surprise'], digit)
            f_neutral = round(face_result['emotion']['neutral'], digit)
            timestamp = origin_file_name.split('_')[1]
            text = f'{char_name},{timestamp},{f_angry},{f_disgust},{f_fear},{f_happy},{f_sad},{f_surprise},{f_neutral}\n'
            f.write(text)
            f.close()

        font_scale = width / ratio_font
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

    name_emotion_file = f"{path_output}/{origin_file_name}.png"
    cv2.imwrite(name_emotion_file, img)
    print(f'{name_emotion_file} generated.')


class Emotion:
    def __init__(self, array_emotion, timestamp):
        self.array_emotion = array_emotion
        self.timestamp = timestamp


class Character:
    def __init__(self, name, gender, profile):
        self.name = name
        self.gender = gender
        self.profile = profile
        self.list_emotion = []


if __name__ == "__main__":
    # 인물 프로필
    path_profile = './the_man_from_nowhere/profile/'
    list_profile = [
        Character('Cha Tae Sik 1', GENDER_MALE, path_profile + 'won_bin_before.png'),
        Character('Cha Tae Sik 2', GENDER_MALE, path_profile + 'won_bin_after.png'),
        Character('Jeong So Mi', GENDER_FEMALE, path_profile + 'kim_sae_ron.png'),
        Character('Man Seok', GENDER_MALE, path_profile + 'kim_hie_won.png'),
        Character('Jong Seok', GENDER_MALE, path_profile + 'kim_seung_o.png'),
        Character('Lam Loan', GENDER_MALE, path_profile + 'thanayong_wongtrakun.png'),
    ]

    # 데이터 생성
    list_shot = natsort.natsorted(os.listdir("./the_man_from_nowhere/indexed_shot"))
    for shot in list_shot:
        identify_character(list_profile, f'./the_man_from_nowhere/indexed_shot/{shot}',
                           './the_man_from_nowhere/shot_emotion',
                           threshold_dist=0.86)

    """
    # 데이터 읽기
    file_notting_hill = open('./notting_hill/shot_emotion/emotion.txt', 'r')
    while True:
        line = file_notting_hill.readline()
        if not line:
            break
        list_data = line.split(',')
        list_data[8] = list_data[8].split('\n')[0]
        for profile in list_profile_notting_hill:
            if profile.name == list_data[0]:
                profile.list_emotion.append(Emotion(list_data[2:9], int(list_data[1])))
    file_notting_hill.close()
    """
