import os

import cv2
import natsort
from deepface import DeepFace


def get_most_common(list_target: list):
    from collections import Counter
    word_counts = Counter(list_target)
    # 가장 많이 나온 값을 출력
    result = word_counts.most_common(1)
    if len(result) > 0:
        return result[0][0]
    else:
        return None


def identify_character(path_img: str, dir_profile: str, path_output: str, flag_mask=False, flag_preprocess=False):
    models: list[str] = [
        "VGG-Face",
        # "Facenet",
        # "Facenet512",
        # "OpenFace",
        # "DeepFace",
        # "ArcFace",
    ]
    backends_analyze: list[str] = [
        # 'opencv',
        # 'ssd',
        # 'mtcnn',
        'retinaface',
        # 'mediapipe',
    ]
    backends_find: list[str] = [
        'opencv',
        # 'ssd',
        # 'mtcnn',
        # 'retinaface',
        # 'mediapipe',
    ]
    # 감정 목록
    EMOTION: list[str] = [
        'angry',
        'disgust',
        'fear',
        'happy',
        'sad',
        'surprise',
        'neutral'
    ]
    aligns: list[bool] = [
        # False,
        True
    ]
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
    len_max_emotion = len(max(EMOTION, key=len))
    path_array = path_img.split("/")
    dot_array = path_array[-1].split(".")
    origin_file_name = dot_array[0]
    METHOD_DISTANCE = [
        # ('cosine', 0.4),
        # ('euclidean', 0.6),
        ('euclidean_l2', 0.86)
    ]
    frame = cv2.imread(path_img)
    if flag_preprocess:
        smoothed = cv2.GaussianBlur(frame, (9, 9), 10)
        frame = cv2.addWeighted(frame, 1.5, smoothed, -0.5, 0)
    frame_save = cv2.imread(path_img)
    for backend_analyze in backends_analyze:
        result_face_analyze = DeepFace.analyze(img_path=path_img,
                                               actions=('gender', 'emotion', 'race', 'age'),
                                               detector_backend=backend_analyze, enforce_detection=False, silent=True)
        for face_index, face_result in enumerate(result_face_analyze):
            list_result: list[str] = []
            # roi
            point_x, point_y, width, height = face_result['region']['x'], face_result['region']['y'], \
                                              face_result['region']['w'], face_result['region']['h']
            if width < 50 or width > 1500:
                print(f"{path_img} - {face_index} - {backend_analyze} - The face is too small or large.")
                break
            frame_face = frame[point_y:point_y + height, point_x:point_x + width]
            for backend_find in backends_find:
                for model in models:
                    for metric in METHOD_DISTANCE:
                        for flag_align in aligns:
                            try:
                                dfs = DeepFace.find(img_path=frame_face, db_path=dir_profile,
                                                    detector_backend=backend_find,
                                                    model_name=model, enforce_detection=False, silent=True,
                                                    align=flag_align,
                                                    distance_metric=metric[0])
                                result = dfs[0].to_dict(orient='list')
                                name_prop = f"{model}_{metric[0]}"
                                dist = result[name_prop][0]
                                name_identity: str = result['identity'][0]
                                print(
                                    f"{path_img} - {face_index} - {backend_analyze} - "
                                    f"{backend_find} - {model} - {metric[0]} - align({flag_align}) - "
                                    f"{name_identity} - {result[name_prop][0]} | {metric[1]}")
                                name_dir = name_identity.split('\\')[1].split('/')[0]
                                if flag_mask and dist > metric[1]:
                                    print(f"{name_dir} not appended")
                                else:
                                    print(f"{name_dir} appended")
                                    list_result.append(name_dir)
                            except Exception as e:
                                print(e)
                                print(
                                    f"{path_img} - {face_index} - {backend_analyze} - {backend_find} - "
                                    f"{model} - {metric[0]} - align({flag_align}) - not found")
            if len(list_result) > 0:
                name_recognized = get_most_common(list_result)
                print(f"\n[결과] {path_img} - {face_index} - {name_recognized}\n")
            else:
                name_recognized = ""
                print(f"\n[결과] {path_img} - {face_index} - unknown\n")
            name_txt_file = f'{path_output}/emotion.csv'
            if os.path.isfile(name_txt_file):
                f = open(name_txt_file, 'a')
            else:
                f = open(name_txt_file, 'w')
                header = 'name,timestamp'
                for e in EMOTION:
                    header += f",{e}"
                header += "\n"
                f.write(header)
            list_fe = []
            for e in EMOTION:
                list_fe.append(round(face_result['emotion'][e], 1))
            timestamp = origin_file_name.split('_')[1]
            text = f'{name_recognized},{timestamp},{list_fe[0]},{list_fe[1]},{list_fe[2]},' \
                   f'{list_fe[3]},{list_fe[4]},{list_fe[5]},{list_fe[6]}\n'
            f.write(text)
            f.close()
            font_scale = width / 350
            size, base_line = cv2.getTextSize("Sample Text For Test", font_face, font_scale, font_thickness)
            offset_y = size[1]
            cv2.rectangle(frame_save, (point_x, point_y), (point_x + width, point_y + height), (128, 128, 0), 2)
            if name_recognized != "":
                content = name_recognized
            else:
                content = str(face_index + 1)
            cv2.putText(frame_save, content,
                        (point_x + offset_x, point_y + offset_y + margin_y),
                        font_face,
                        font_scale,
                        color_font_base,
                        font_thickness)
            for emotion_index, emotion in enumerate(EMOTION):
                if face_result['dominant_emotion'] == emotion:
                    cv2.putText(frame_save,
                                f"{emotion.ljust(len_max_emotion)} {round(face_result['emotion'][emotion], digit)}",
                                (point_x + offset_x, point_y + offset_y * (emotion_index + 2) + margin_y),
                                font_face,
                                font_scale,
                                (0, 255, 0), font_thickness)
                else:
                    cv2.putText(frame_save,
                                f"{emotion.ljust(len_max_emotion)} {round(face_result['emotion'][emotion], digit)}",
                                (point_x + offset_x, point_y + offset_y * (emotion_index + 2) + margin_y),
                                font_face,
                                font_scale,
                                color_font_base, font_thickness)
    name_emotion_file = f"{path_output}/{origin_file_name}.png"
    cv2.imwrite(name_emotion_file, frame_save)
    print(f'{name_emotion_file} generated.')


if __name__ == "__main__":
    path_main = './the_man_from_nowhere'
    path_result = f'{path_main}/shot_emotion'
    if not os.path.exists(path_result):
        os.makedirs(path_result)
    path_indexed = f'{path_main}/indexed_shot'
    list_shot = natsort.natsorted(os.listdir(path_indexed))
    path_profile = f"{path_main}/profile"

    for shot in list_shot:
        identify_character(f'{path_indexed}/{shot}', path_profile, path_result, flag_mask=False, flag_preprocess=True)
