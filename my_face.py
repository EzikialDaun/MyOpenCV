import copy
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from deepface import DeepFace
import natsort

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


def identify_character(profiles, img_name, path_output, threshold_dist=0.5):
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
        delete_index = None
        for p_index, profile in enumerate(copied_profiles):
            verify_result = DeepFace.verify(img1_path=profile.profile, img2_path=img_frag,
                                            enforce_detection=False,
                                            detector_backend='retinaface', model_name='Facenet',
                                            distance_metric='cosine')
            print(verify_result)
            if verify_result['distance'] <= threshold_dist:
                char_name = profile.name
                delete_index = p_index
                for origin_profile in profiles:
                    if origin_profile.name == char_name:
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
                        break
                break
        if delete_index is not None:
            copied_profiles.pop(delete_index)

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
    path_profile_notting_hill = './notting_hill/profile/'
    list_profile_notting_hill = [
        Character('William Thacker', GENDER_MALE, path_profile_notting_hill + 'hugh_grant.png'),
        Character('Anna Scott', GENDER_FEMALE, path_profile_notting_hill + 'julia_roberts.png'),
        Character('Spike', GENDER_MALE, path_profile_notting_hill + 'rhys_ifans.png'),
        Character('Bernie', GENDER_MALE, path_profile_notting_hill + 'hugh_bonneville.png'),
        Character('Bella', GENDER_FEMALE, path_profile_notting_hill + 'gina_mckee.png'),
        Character('Honey', GENDER_FEMALE, path_profile_notting_hill + 'emma_chambers.jpg'),
        Character('Max', GENDER_MALE, path_profile_notting_hill + 'tim_mcinnerny.jpg')
    ]

    path_profile_roman_holiday = './roman_holiday/profile/'
    list_profile_roman_holiday = [
        Character('Anne', GENDER_FEMALE, path_profile_roman_holiday + 'audrey_hepburn.png'),
        Character('Joe Bradley', GENDER_MALE, path_profile_roman_holiday + 'gregory_peck.png'),
        Character('Irving Radovich', GENDER_MALE, path_profile_roman_holiday + 'eddie_albert.png')
    ]

    """
    # 데이터 생성
    list_shot = natsort.natsorted(os.listdir("./roman_holiday/indexed_shot"))
    for shot in list_shot:
        identify_character(list_profile, f'./roman_holiday/indexed_shot/{shot}', './roman_holiday/shot_emotion',
                           threshold_dist=0.65)
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

    file_roman_holiday = open('./roman_holiday/shot_emotion/emotion.txt', 'r')
    while True:
        line = file_roman_holiday.readline()
        if not line:
            break
        list_data = line.split(',')
        list_data[8] = list_data[8].split('\n')[0]
        for profile in list_profile_roman_holiday:
            if profile.name == list_data[0]:
                profile.list_emotion.append(Emotion(list_data[2:9], int(list_data[1])))
    file_roman_holiday.close()

    list_will_happy = []
    list_will_angry = []
    list_will_sad = []

    for emotion in list_profile_notting_hill[0].list_emotion:
        list_will_happy.append(emotion.array_emotion[3])
        list_will_angry.append(emotion.array_emotion[0])
        list_will_sad.append(emotion.array_emotion[4])

    list_joe_happy = []
    list_joe_angry = []
    list_joe_sad = []

    for emotion in list_profile_roman_holiday[1].list_emotion:
        list_joe_happy.append(emotion.array_emotion[3])
        list_joe_angry.append(emotion.array_emotion[0])
        list_joe_sad.append(emotion.array_emotion[4])

    list_anna_happy = []
    list_anna_angry = []
    list_anna_sad = []

    for emotion in list_profile_notting_hill[1].list_emotion:
        list_anna_happy.append(emotion.array_emotion[3])
        list_anna_angry.append(emotion.array_emotion[0])
        list_anna_sad.append(emotion.array_emotion[4])

    list_anne_happy = []
    list_anne_angry = []
    list_anne_sad = []

    for emotion in list_profile_roman_holiday[0].list_emotion:
        list_anne_happy.append(emotion.array_emotion[3])
        list_anne_angry.append(emotion.array_emotion[0])
        list_anne_sad.append(emotion.array_emotion[4])

    import numpy as np
    from scipy.interpolate import interp1d

    # 선형 보간 함수 생성
    interpolator_joe_happy = interp1d(np.linspace(0, 1, len(list_joe_happy)), list_joe_happy, kind='linear',
                                      fill_value='extrapolate')
    interpolator_joe_angry = interp1d(np.linspace(0, 1, len(list_joe_angry)), list_joe_angry, kind='linear',
                                      fill_value='extrapolate')
    interpolator_joe_sad = interp1d(np.linspace(0, 1, len(list_joe_sad)), list_joe_sad, kind='linear',
                                    fill_value='extrapolate')
    interpolator_anne_happy = interp1d(np.linspace(0, 1, len(list_anne_happy)), list_anne_happy, kind='linear',
                                       fill_value='extrapolate')
    interpolator_anne_angry = interp1d(np.linspace(0, 1, len(list_anne_angry)), list_anne_angry, kind='linear',
                                       fill_value='extrapolate')
    interpolator_anne_sad = interp1d(np.linspace(0, 1, len(list_anne_sad)), list_anne_sad, kind='linear',
                                     fill_value='extrapolate')

    # 보간된 데이터 생성
    interpolated_joe_happy = interpolator_joe_happy(np.linspace(0, 1, len(list_will_happy)))
    interpolated_joe_angry = interpolator_joe_angry(np.linspace(0, 1, len(list_will_angry)))
    interpolated_joe_sad = interpolator_joe_sad(np.linspace(0, 1, len(list_will_sad)))
    interpolated_anne_happy = interpolator_anne_happy(np.linspace(0, 1, len(list_anna_happy)))
    interpolated_anne_angry = interpolator_anne_angry(np.linspace(0, 1, len(list_anna_angry)))
    interpolated_anne_sad = interpolator_anne_sad(np.linspace(0, 1, len(list_anna_sad)))

    from sklearn.preprocessing import MinMaxScaler

    # 데이터를 0과 1 사이의 값으로 정규화
    scaler = MinMaxScaler()

    normalized_joe_happy = scaler.fit_transform(interpolated_joe_happy.reshape(-1, 1)).flatten()
    normalized_will_happy = scaler.fit_transform(np.array(list_will_happy).reshape(-1, 1)).flatten()
    normalized_joe_angry = scaler.fit_transform(interpolated_joe_angry.reshape(-1, 1)).flatten()
    normalized_will_angry = scaler.fit_transform(np.array(list_will_angry).reshape(-1, 1)).flatten()
    normalized_joe_sad = scaler.fit_transform(interpolated_joe_sad.reshape(-1, 1)).flatten()
    normalized_will_sad = scaler.fit_transform(np.array(list_will_sad).reshape(-1, 1)).flatten()
    normalized_anne_happy = scaler.fit_transform(interpolated_anne_happy.reshape(-1, 1)).flatten()
    normalized_anna_happy = scaler.fit_transform(np.array(list_anna_happy).reshape(-1, 1)).flatten()
    normalized_anne_angry = scaler.fit_transform(interpolated_anne_angry.reshape(-1, 1)).flatten()
    normalized_anna_angry = scaler.fit_transform(np.array(list_anna_angry).reshape(-1, 1)).flatten()
    normalized_anne_sad = scaler.fit_transform(interpolated_anne_sad.reshape(-1, 1)).flatten()
    normalized_anna_sad = scaler.fit_transform(np.array(list_anna_sad).reshape(-1, 1)).flatten()

    # 정규화된 데이터 간의 상관 계수 계산
    correlation_coefficient_male_happy = np.corrcoef(normalized_joe_happy, normalized_will_happy)[0, 1]
    correlation_coefficient_male_angry = np.corrcoef(normalized_joe_angry, normalized_will_angry)[0, 1]
    correlation_coefficient_male_sad = np.corrcoef(normalized_joe_sad, normalized_will_sad)[0, 1]

    print("윌리엄과 조")
    print(f"'행복' 감정 값 변화의 상관 계수: {correlation_coefficient_male_happy}")
    print(f"'분노' 감정 값 변화의 상관 계수: {correlation_coefficient_male_angry}")
    print(f"'슬픔' 감정 값 변화의 상관 계수: {correlation_coefficient_male_sad}")
    print()

    correlation_coefficient_female_happy = np.corrcoef(normalized_anne_happy, normalized_anna_happy)[0, 1]
    correlation_coefficient_female_angry = np.corrcoef(normalized_anne_angry, normalized_anna_angry)[0, 1]
    correlation_coefficient_female_sad = np.corrcoef(normalized_anne_sad, normalized_anna_sad)[0, 1]

    print("안나와 앤")
    print(f"'행복' 감정 값 변화의 상관 계수: {correlation_coefficient_female_happy}")
    print(f"'분노' 감정 값 변화의 상관 계수: {correlation_coefficient_female_angry}")
    print(f"'슬픔' 감정 값 변화의 상관 계수: {correlation_coefficient_female_sad}")

    list_male_x = list(range(0, len(list_will_sad)))
    list_female_x = list(range(0, len(list_anna_sad)))

    plt.plot(list_male_x, normalized_will_happy)
    plt.plot(list_male_x, normalized_joe_happy)
    plt.legend(['will_happy', 'joe_happy'])
    plt.title('male_happy')
    plt.show()

    plt.clf()

    plt.plot(list_male_x, normalized_will_angry)
    plt.plot(list_male_x, normalized_joe_angry)
    plt.legend(['will_angry', 'joe_angry'])
    plt.title('male_angry')
    plt.show()

    plt.clf()

    plt.plot(list_male_x, normalized_will_sad)
    plt.plot(list_male_x, normalized_joe_sad)
    plt.legend(['will_sad', 'joe_sad'])
    plt.title('male_sad')
    plt.show()

    plt.clf()

    plt.plot(list_female_x, normalized_anna_happy)
    plt.plot(list_female_x, normalized_anne_happy)
    plt.legend(['anna_happy', 'anne_happy'])
    plt.title('female_happy')
    plt.show()

    plt.clf()

    plt.plot(list_female_x, normalized_anna_angry)
    plt.plot(list_female_x, normalized_anne_angry)
    plt.legend(['anna_angry', 'anne_angry'])
    plt.title('female_angry')
    plt.show()

    plt.clf()

    plt.plot(list_female_x, normalized_anna_sad)
    plt.plot(list_female_x, normalized_anne_sad)
    plt.legend(['anna_sad', 'anne_sad'])
    plt.title('female_sad')
    plt.show()

    plt.clf()
