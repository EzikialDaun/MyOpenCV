import csv

import cv2
from deepface import DeepFace

from MyOpenCV.get_dominate_haircolor import extract_hair_area, has_black_hair, detect_hat


def select_similar(list_target: list[list]):
    result = {}
    for key, value in list_target:
        if key not in result or value < result[key]:
            result[key] = value

    # 결과를 다시 리스트 형태로 변환
    return [[key, value] for key, value in result.items()]


def identify_char(dir_profile: str, dir_probe: str, path_label: str, path_profile):
    distance_expand = 0.3
    metrics = ["cosine", "euclidean", "euclidean_l2"]
    models = [
        "VGG-Face",
        "Facenet",
        "Facenet512",
        "DeepFace",
        "ArcFace",
    ]
    backends = [
        'opencv',
        'mtcnn',
        'retinaface',
    ]
    weight_race = 1
    weight_gender = 1
    weight_hair = 1
    weight_hat = 1

    f = open(path_label, 'r')
    rdr = csv.reader(f)
    list_label = []
    for line in rdr:
        list_label.append([line[0], line[1]])
    # 헤더 제거
    list_label = list_label[1:]
    f.close()

    f = open(path_profile, 'r')
    rdr = csv.reader(f)
    list_profile = []
    for line in rdr:
        list_profile.append([line[0], line[1], line[2], line[3], line[4]])
    # 헤더 제거
    list_profile = list_profile[1:]
    f.close()

    score = 0

    for k, line in enumerate(list_label):
        target = f"{dir_probe}\\{line[0]}"
        src = cv2.imread(target)
        analyzed = DeepFace.analyze(
            img_path=src,
            actions=['gender', 'race'],
            detector_backend=backends[2],
            enforce_detection=False,
            silent=True
        )
        from deepface.modules import verification
        # 쓰레스홀드 확장하여 후보를 더 추출
        found = DeepFace.find(img_path=src, db_path=dir_profile,
                              detector_backend=backends[2], model_name=models[2], distance_metric=metrics[2],
                              enforce_detection=False,
                              threshold=verification.find_threshold(models[2], metrics[2]) + distance_expand,
                              silent=True)
        race = analyzed[0]['dominant_race']
        gender = analyzed[0]['dominant_gender']
        try:
            has_hat = detect_hat(src)
        except IndexError:
            has_hat = None
        if has_hat == "1":
            is_black_hair = None
        else:
            try:
                x, y, w, h = (analyzed[0]["region"][key] for key in ("x", "y", "w", "h"))
                padding = int((w + h) / 4)

                y_start = max(0, y - padding)
                y_end = min(src.shape[0], y + int((h + padding) / 2))
                # y_end = min(src.shape[1], y + h + padding)
                x_start = max(0, x - padding)
                x_end = min(src.shape[1], x + w + padding)

                frag = src[y_start:y_end, x_start:x_end]

                colors, seg = extract_hair_area(frag)

                is_black_hair = has_black_hair(colors)
            except IndexError:
                is_black_hair = None

        print(f"race : {race}")
        print(f"gender : {gender}")
        print(f"is_black_hair : {is_black_hair}")
        print(f"has_hat : {has_hat}")

        list_distance = []
        for i in found[0].values:
            # name: str = i[0].split('\\')[6]
            import os
            name = os.path.basename(os.path.dirname(i[0]))
            distance: float = i[11]
            list_distance.append([name, distance])
        # list_selected = select_similar(list_distance)
        list_selected = sorted(select_similar(list_distance), key=lambda x: x[1])
        print(list_selected)
        for i in list_selected:
            for j in list_profile:
                if i[0] == j[0]:
                    if race == j[1]:
                        i[1] *= weight_race
                    if gender == j[2]:
                        i[1] *= weight_gender
                    if is_black_hair == j[3]:
                        i[1] *= weight_hair
                    if has_hat == j[4]:
                        i[1] *= weight_hat
                    break
        sorted_data = sorted(list_selected, key=lambda x: x[1])
        print(sorted_data)

        answer = sorted_data[0][0]
        print(f"target: {target}, label: {line[1]}, answer: {answer}")
        if answer == line[1]:
            score += 1
        print(f"{score} / {k + 1} = {score / (k + 1)}")
        print()


if __name__ == "__main__":
    movie = "..\\..\\lab\\MyFace Dataset Lite\\inception"
    label = f"{movie}\\label.csv"
    probe = f"{movie}\\probe"
    profiles = f"{movie}\\profile"
    profile = f"{movie}\\profile.csv"
    identify_char(profiles, probe, label, profile)
