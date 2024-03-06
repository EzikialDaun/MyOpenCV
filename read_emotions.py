from MyOpenCV.get_emotions import Character, Emotion

if __name__ == "__main__":
    path_main = './the_man_from_nowhere'
    list_name = ['Cha Tae Sik 1', 'Cha Tae Sik 2', 'Jeong So Mi', 'Man Seok', 'Jong Seok', 'Lam Loan']
    list_profile = []
    for name in list_name:
        list_profile.append(Character(name, f'{path_main}/profile/{name.replace(" ", "_").lower()}'))
    file = open(f'{path_main}/shot_emotion_el2/emotion.csv', 'r')
    len_data = 9
    while True:
        line = file.readline()
        if not line:
            break
        list_data = line.split(',')
        if len(list_data) != len_data:
            continue
        list_data[len_data - 1] = list_data[len_data - 1].split('\n')[0]
        for profile in list_profile:
            if profile.name == list_data[0].strip():
                profile.list_emotion.append(Emotion(list_data[2:len_data], int(list_data[1])))
    file.close()
    print(list_profile)
