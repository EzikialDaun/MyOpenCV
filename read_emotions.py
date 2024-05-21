import os
import pandas as pd


def get_dir_names(path_dir: str) -> list[str]:
    if os.path.exists(path_dir):
        # 지정한 경로에 있는 모든 디렉토리의 이름을 문자열 리스트로 저장
        return [d for d in os.listdir(path_dir) if os.path.isdir(os.path.join(path_dir, d))]
    else:
        return []


if __name__ == "__main__":
    dir_main = './the_man_from_nowhere'
    dir_profile = f"{dir_main}/profile"
    path_csv = f"{dir_main}/shot_emotion_el2_retinaface/emotion.csv"
    list_name = get_dir_names(dir_profile)
    df = pd.read_csv(path_csv).values.tolist()
    print(df)
