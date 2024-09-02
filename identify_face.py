import csv

from deepface import DeepFace


def calculate_match_ratio(set1, set2):
    # 교집합의 크기를 구합니다.
    intersection_size = len(set1.intersection(set2))

    # set1의 크기를 구합니다.
    set1_size = len(set1)

    # 일치하는 비율을 계산합니다.
    match_ratio = intersection_size / set1_size

    return match_ratio


def identify_face(path_label: str):
    metrics = ["cosine", "euclidean", "euclidean_l2"]
    models = [
        "VGG-Face",
        "Facenet",
        "Facenet512",
        "DeepFace",
        "DeepID",
        "ArcFace",
        "SFace",
        "GhostFaceNet",
    ]
    backends = [
        'opencv',
        'mtcnn',
        'fastmtcnn',
        'retinaface',
        'yunet',
        'centerface',
    ]

    f = open(path_label, 'r')
    rdr = csv.reader(f)

    list_label = []
    for line in rdr:
        list_label.append([line[0], line[1].split('|'), line[2].split('|')])
    # 헤더 제거
    list_label = list_label[1:]

    f.close()

    len_side = 0
    len_frontal = 0

    for item in list_label:
        if "F" in item[2]:
            len_side += 1
        else:
            len_frontal += 1

    print(f"frontal len : {len_frontal}")
    print(f"side len : {len_side}")

    sum_side_score = 0
    sum_frontal_score = 0

    for index, label in enumerate(list_label):
        dfs = DeepFace.find(img_path=f"../../MyFace Dataset/probe/{label[0]}.png",
                            db_path="../../MyFace Dataset/gallery",
                            detector_backend=backends[1], model_name=models[5], distance_metric=metrics[2],
                            enforce_detection=False)

        list_answer = []
        for data in dfs:
            if len(data) > 0:
                list_answer.append(data.values[0][0].split('\\')[1].split('/')[0])

        score = calculate_match_ratio(set(label[1]), set(list_answer))
        print(f"label - {set(label[1])}")
        print(f"answer - {set(list_answer)}")

        is_frontal = "F" in label[2]
        if is_frontal:
            sum_side_score += score
        else:
            sum_frontal_score += score

        print(f"{label[0]}.png : {score} - is_frontal({is_frontal})")
        print(f"frontal accuracy : {sum_frontal_score / len_frontal}")
        print(f"side accuracy : {sum_side_score / len_side}")
        print()


if __name__ == "__main__":
    identify_face('../../MyFace Dataset/label.csv')
