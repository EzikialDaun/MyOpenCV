import cv2


def get_shots(path_input, path_output, alpha=1.0, limit=-1, interval=1):
    capture = cv2.VideoCapture(path_input)
    prev_hist = None
    cnt_shot = 0

    # 최초 프레임
    first_ret, first_frame = capture.read()
    if first_ret:
        prev_hist = cv2.calcHist([first_frame], [0], None, [256], [0, 256])
        cv2.imwrite(f'{path_output}/{cnt_shot}_0.png', first_frame)
        print(f'{path_output}/{cnt_shot}_0.png')
        print()
        cnt_shot += 1
    else:
        return

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        if limit != -1 and cnt_shot >= limit:
            break

        if capture.get(cv2.CAP_PROP_POS_FRAMES) % interval == 0:
            # 샷 체인지 검출
            curr_hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
            # 0: 완전 일치, 1: 완전 불일치
            diff = cv2.compareHist(curr_hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA)
            print(f'diff: {diff}')
            if diff >= alpha:
                file_name = f'{path_output}/{cnt_shot}_{int(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)}.png'
                cv2.imwrite(file_name, frame)
                print(file_name)
                print()
                cnt_shot += 1
                prev_hist = curr_hist

    # 종료
    if capture.isOpened():
        # 사용한 자원 해제
        capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    get_shots(path_input='./data/the_man_from_nowhere.mkv', path_output='./the_man_from_nowhere/indexed_shot',
              interval=24,
              alpha=0.25)
