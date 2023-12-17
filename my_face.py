import copy
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace

# 데이터 디렉토리 경로
path_data = './data/'

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


def identify_character(profiles, img_name):
    face_analyze_result = DeepFace.analyze(img_path=img_name,
                                           actions=('gender', 'emotion'),
                                           detector_backend='retinaface', enforce_detection=False)
    img = cv2.imread(img_name)

    # 전처리
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
            if verify_result['distance'] <= 0.5:
                char_name = profile.name
                delete_index = p_index
                for origin_profile in profiles:
                    if origin_profile.name == profile.name:
                        origin_profile.list_emotion.append(Emotion(round(face_result['emotion']['angry'], digit),
                                                                   round(face_result['emotion']['disgust'], digit),
                                                                   round(face_result['emotion']['fear'], digit),
                                                                   round(face_result['emotion']['happy'], digit),
                                                                   round(face_result['emotion']['sad'], digit),
                                                                   round(face_result['emotion']['surprise'], digit),
                                                                   round(face_result['emotion']['neutral'], digit)))
                        break
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
    cv2.imwrite(f"./notting_hill/shot_emotion/{origin_file_name}_retina.png", img)
    # 프로그램 종료 방지, 아무 키 누르면 프로그램 종료
    # cv2.waitKey()
    # 모든 창 소멸
    # cv2.destroyAllWindows()


class Emotion:
    def __init__(self, angry, disgust, fear, happy, sad, surprise, neutral):
        self.angry = angry
        self.disgust = disgust
        self.fear = fear
        self.happy = happy
        self.sad = sad
        self.surprise = surprise
        self.neutral = neutral


class Character:
    def __init__(self, name, profile):
        self.name = name
        self.profile = profile
        self.list_emotion = []


def get_shots(path_input, alpha, limit=-1, interval=-1):
    capture = cv2.VideoCapture(path_input)
    prev_hist = None
    cnt_global = 0
    cnt_shot = 0
    interval_frame = 0
    # 샷 변경을 검출하기 위해 비교하는 임의의 기준값 alpha
    if interval == -1:
        interval_frame = int(1000 / capture.get(cv2.CAP_PROP_FPS))
    else:
        interval_frame = interval
    print(f'interval_frame: {interval_frame}')
    print()

    # 최초 프레임
    first_ret, first_frame = capture.read()
    if first_ret:
        prev_hist = cv2.calcHist([first_frame], [0], None, [256], [0, 256])
        cv2.imwrite(f'./shots/{cnt_shot}.png', first_frame)
        print(f'./shots/{cnt_shot}.png')
        print()
        cnt_shot += 1
        cnt_global += 1

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        if limit != -1 and cnt_shot >= limit:
            break

        if cnt_global % interval_frame == 0:
            # 샷 체인지 검출
            curr_hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
            diff = cv2.compareHist(curr_hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff >= alpha:
                cv2.imwrite(f'./shots/{cnt_shot}.png', frame)
                print(f'diff: {diff}')
                print(f'./shots/{cnt_shot}.png')
                print()
                cnt_shot += 1
                prev_hist = curr_hist

        cnt_global += 1

    # 종료
    if capture.isOpened():
        # 사용한 자원 해제
        capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    path_profile = './notting_hill/profile/'
    list_profile = [
        Character('William Thacker', path_profile + 'hugh_grant.png'),
        Character('Anna Scott', path_profile + 'julia_roberts.png'),
        Character('Spike', path_profile + 'rhys_ifans.png'),
        Character('Bernie', path_profile + 'hugh_bonneville.png'),
        Character('Bella', path_profile + 'gina_mckee.png'),
        Character('Honey', path_profile + 'emma_chambers.jpg'),
        Character('Max', path_profile + 'tim_mcinnerny.jpg')
    ]

    list_shot = os.listdir("./notting_hill/shot")
    for shot in list_shot:
        identify_character(list_profile, f'./notting_hill/shot/{shot}')

    for person in list_profile:
        sum_angry = 0
        sum_disgust = 0
        sum_fear = 0
        sum_happy = 0
        sum_sad = 0
        sum_surprise = 0
        sum_neutral = 0

        len_emotion = len(person.list_emotion)
        if len_emotion <= 0:
            break

        for emotion in person.list_emotion:
            sum_angry += emotion.angry
            sum_disgust += emotion.disgust
            sum_fear += emotion.fear
            sum_happy += emotion.happy
            sum_sad += emotion.sad
            sum_surprise += emotion.surprise
            sum_neutral += emotion.neutral

        avg_angry = sum_angry / len_emotion
        avg_disgust = sum_disgust / len_emotion
        avg_fear = sum_fear / len_emotion
        avg_happy = sum_happy / len_emotion
        avg_sad = sum_sad / len_emotion
        avg_surprise = sum_surprise / len_emotion
        avg_neutral = sum_neutral / len_emotion

        values = [
            avg_angry,
            avg_disgust,
            avg_fear,
            avg_happy,
            avg_sad,
            avg_surprise,
            avg_neutral
        ]

        x = np.arange(len(list_emotion))

        plt.bar(x, values)
        plt.xticks(x, list_emotion)
        plt.title(f'{person.name} - graph - {len_emotion} shots')
        plt.savefig(f'./notting_hill/graph_emotion/{person.name}_graph.png')
        plt.clf()
