import pandas as pd

from MyOpenCV.get_emotions import Character, Emotion, list_emotion

if __name__ == "__main__":
    path_main = './the_man_from_nowhere'
    list_name = ['Cha Tae Sik', 'Jeong So Mi', 'Man Seok', 'Jong Seok', 'Lam Loan']
    list_profile = []
    for name in list_name:
        list_profile.append(Character(name, f'{path_main}/profile/{name.replace(" ", "_").lower()}'))

    data = pd.read_csv(f'{path_main}/shot_emotion_el2/emotion.csv')
    for idx, row in data.iterrows():
        for profile in list_profile:
            if row['name'].strip() == profile.name:
                list_temp = []
                for e in list_emotion:
                    list_temp.append(row[e])
                profile.list_emotion.append(Emotion(list_temp, row['timestamp']))
    print(list_profile)
