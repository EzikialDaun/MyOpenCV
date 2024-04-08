import os
import cv2
from deepface import DeepFace
from natsort import natsort
from ultralytics import YOLO

from MyOpenCV.get_dominant_color import get_dominant_colors
from MyOpenCV.get_emotions import Character

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


def identify_character(profiles: list[Character], img_name: str, path_output: str):
    method_distance = [('cosine', 0.4), ('euclidean', 0.6), ('euclidean_l2', 0.86)]
    result = model.predict(img_name, save=False)
    frame = cv2.imread(img_name)
    for i in range(len(result[0].boxes)):
        list_coord = result[0].boxes[i].xyxy[0].tolist()
        coord_start = (int(list_coord[0]), int(list_coord[1]))
        coord_end = (int(list_coord[2]), int(list_coord[3]))
        shoulder_left = result[0].keypoints.xy[i][6].tolist()
        shoulder_right = result[0].keypoints.xy[i][5].tolist()
        coord_shoulder_left = (int(shoulder_left[0]), int(shoulder_left[1]))
        coord_shoulder_right = (int(shoulder_right[0]), int(shoulder_right[1]))
        is_damaged_left = False
        is_damaged_right = False
        for coord in shoulder_left:
            if coord == 0:
                is_damaged_left = True
        for coord in shoulder_right:
            if coord == 0:
                is_damaged_right = True
        if not is_damaged_left and not is_damaged_right:
            size_box = 50
            if not is_damaged_left:
                coord_shd_start = (
                    coord_shoulder_left[0] - size_box // 2, coord_shoulder_left[1] - size_box // 2)
                coord_shd_end = (
                    coord_shoulder_left[0] + size_box // 2, coord_shoulder_left[1] + size_box // 2)
            else:
                coord_shd_start = (
                    coord_shoulder_right[0] - size_box // 2, coord_shoulder_right[1] - size_box // 2)
                coord_shd_end = (
                    coord_shoulder_right[0] + size_box // 2, coord_shoulder_right[1] + size_box // 2)
            frame_frag = frame[coord_start[1]:coord_end[1], coord_start[0]:coord_end[0]]
            result_analyze = DeepFace.analyze(
                img_path=frame_frag,
                actions=('gender', 'emotion', 'age', 'race'),
                detector_backend='retinaface', enforce_detection=False,
                silent=True)
            frame_shoulder = frame[coord_shd_start[1]:coord_shd_end[1], coord_shd_start[0]:coord_shd_end[0]]
            color_clothing = get_dominant_colors(frame_shoulder, 1)
            rgb_clothing = (color_clothing[0][0], color_clothing[0][1], color_clothing[0][2])
            if len(result_analyze) > 0:
                point_x, point_y, width, height = result_analyze[0]['region']['x'], result_analyze[0]['region'][
                    'y'], result_analyze[0]['region']['w'], result_analyze[0]['region']['h']
                frame_bound = frame_frag[point_y:point_y + height, point_x:point_x + width]
                for profile in profiles:
                    result_find = DeepFace.find(img_path=frame_bound, db_path=profile.profile, enforce_detection=False,
                                                distance_metric=method_distance[2][0], silent=True)


if __name__ == "__main__":
    model = YOLO('yolov8n-pose.pt')

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
