import pandas as pd
from MyOpenCV.get_emotions import Character, Interaction

if __name__ == "__main__":
    path_main = './the_man_from_nowhere'
    list_name = ['Cha Tae Sik', 'Jeong So Mi', 'Man Seok', 'Jong Seok', 'Lam Loan']
    list_profile = []
    for name in list_name:
        list_profile.append(Character(name, f'{path_main}/profile/{name.replace(" ", "_").lower()}'))

    # 주인공 선정을 위해 빈도 분석
    data = pd.read_csv(f'{path_main}/shot_emotion_el2/emotion.csv')
    for idx, row in data.iterrows():
        for profile in list_profile:
            if row['name'].strip() == profile.name:
                profile.list_appearance.append(row['timestamp'])

    # 액션 영화 규칙 & 클리셰
    # 1. 등장 빈도가 가장 많은 인물이 주인공이다.
    # 2. 반동 인물과 그 조력자는 보통 주인공과 상호작용 끝에 퇴장한다.
    # 3. 반동 인물의 조력자가 반동 인물보다 먼저 퇴장한다.
    # 4. 주인공의 시점에서 시작과 끝을 함께 하는 인물이 보통 조력자다.

    if len(list_profile) > 0:
        # 규칙 1: 가장 빈도가 높은 주인공 선정
        protagonist = list_profile[0]
        print(f'{protagonist.name}는(은) 작중 가장 많은 비중을 가진 주동 인물입니다.')
        for profile in list_profile:
            if len(profile.list_appearance) > len(protagonist.list_appearance):
                protagonist = profile
        list_rest = [obj for obj in list_profile if obj.name != protagonist.name]

        # 타임아웃
        timeout = 120
        # 상호작용 시점
        timestamp_base = 0
        # 1 대 1 상호작용 원칙을 지키기 위한 락 플래그
        lock = False
        for idx, row in data.iterrows():
            name = row['name'].strip()
            timestamp = row['timestamp']
            if name == protagonist.name:
                timestamp_base = timestamp
                lock = False
            elif timestamp - timestamp_base <= timeout and lock is False:
                lock = True
                protagonist.list_interaction.append(Interaction(timestamp_base, name))

        list_aid_protagonist = []
        len_interaction = len(protagonist.list_interaction)
        if len_interaction > 0:
            # 규칙 4: 주인공과 처음과 끝에 상호작용하는 사람이 조력자
            name_aid_protagonist = protagonist.list_interaction[0].target
            if name_aid_protagonist == protagonist.list_interaction[-1].target:
                list_aid_protagonist.append(protagonist.list_interaction[0])
                print(f'{name_aid_protagonist}는(은) {protagonist.name}의 처음과 끝을 함께하는 조력자입니다.')
                list_rest = [obj for obj in list_rest if obj.name != name_aid_protagonist]
            # 규칙 2, 3: 주인공에 의한 퇴장 시점의 순서
            min_exit = 0
            antagonist = None
            for profile in list_rest:
                appearance_last = [obj for obj in protagonist.list_interaction if obj.target == profile.name][
                    -1].timestamp
                if appearance_last > min_exit:
                    antagonist = profile
                    min_exit = appearance_last
            if antagonist is not None:
                print(f'{antagonist.name}는(은) {protagonist.name}에 의해 가장 나중에 퇴장한 반동 인물입니다.')
                list_aid_antagonist = [obj for obj in list_rest if obj.name != antagonist.name]
                for i in list_aid_antagonist:
                    print(f'{i.name}는(은) 반동 인물 {antagonist.name}의 조력자입니다.')
