import csv
import os
import cv2
import numpy as np
from deepface import DeepFace

from MyOpenCV.confusion_matrix import get_multiple_class_f1_score
from MyOpenCV.get_dominate_haircolor import detect_hat
from deepface.modules import verification


def select_similar(list_target: list[list]):
    result = {}
    for key, value in list_target:
        if key not in result or value < result[key]:
            result[key] = value

    # 결과를 다시 리스트 형태로 변환
    return [[key, value] for key, value in result.items()]


def identify_char(dir_profile: str, dir_probe: str, path_label: str, path_profile, distance_expand=0.3, weight_race=1.0,
                  weight_gender=1.0, weight_hair=1.0, weight_hat=1.0):
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

    len_matrix = len(list_profile)
    confusion_matrix = np.zeros((len_matrix, len_matrix), dtype=int)
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
        print(analyzed)
        # 쓰레스홀드 확장하여 후보를 더 추출
        found = DeepFace.find(img_path=src, db_path=dir_profile,
                              detector_backend=backends[2], model_name=models[2], distance_metric=metrics[2],
                              enforce_detection=False,
                              threshold=verification.find_threshold(models[2], metrics[2]) + distance_expand,
                              silent=True)
        race = analyzed[0]['dominant_race']
        gender = analyzed[0]['dominant_gender']
        is_black_hair = None
        if weight_hat == 1.0:
            has_hat = None
        else:
            has_hat = detect_hat(src)

        print(f"race : {race}")
        print(f"gender : {gender}")
        print(f"is_black_hair : {is_black_hair}")
        print(f"has_hat : {has_hat}")

        list_distance = []
        for i in found[0].values:
            name = os.path.basename(os.path.dirname(i[0]))
            distance: float = i[11]
            list_distance.append([name, distance])
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

        label = line[1]
        answer = sorted_data[0][0]
        index_answer = -1
        index_label = -1
        for i, profile in enumerate(list_profile):
            if answer == profile[0]:
                index_answer = i
            if label == profile[0]:
                index_label = i

        if index_answer == -1 or index_label == -1:
            print("matrix indexing error")
        else:
            confusion_matrix[index_label][index_answer] += 1

        print(f"target: {target}, label: {label}, answer: {answer}")
        if answer == label:
            score += 1
        print(f"{score} / {k + 1} = {score / (k + 1)}")
        print(confusion_matrix)
        print()

    # 텍스트 파일로 저장하기
    with open("result_identify_char.txt", "a") as file:
        file.write(f"Target: {dir_probe}\n")
        file.write(f"Weight For Race: {weight_race}\n")
        file.write(f"Weight For Gender: {weight_gender}\n")
        file.write(f"Weight For Hat: {weight_hat}\n")
        file.write(f"Confusion_Matrix:\n{confusion_matrix}\n")
        file.write(f"F1 Score: {get_multiple_class_f1_score(confusion_matrix)}\n")
        file.write(f"Accuracy: {score / len_matrix}\n")
        file.write("--------\n")  # 구분선 추가


if __name__ == "__main__":
    dir_movie = f"..\\..\\lab\\MyFace Dataset Lite\\the_greatest_showman"
    weights_race = [0.8]
    weights_gender = [0.8]
    weights_hat = [0.65]

    for h in weights_hat:
        for r in weights_race:
            for g in weights_gender:
                identify_char(f"{dir_movie}\\profile", f"{dir_movie}\\probe", f"{dir_movie}\\label.csv",
                              f"{dir_movie}\\profile.csv", weight_race=r, weight_gender=g, weight_hat=h)
