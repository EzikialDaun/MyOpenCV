import cv2
from matplotlib import pyplot as plt
import numpy as np
import pafy

path_data = './data/'


def _0101():
    image_file = path_data + 'lena.jpg'
    img = cv2.imread(image_file)  # cv2.IMREAD_COLOR
    img2 = cv2.imread(image_file, 0)  # cv2.IMREAD_GRAYSCALE
    cv2.imshow('Lena color', img)
    cv2.imshow('Lena grayscale', img2)

    cv2.waitKey()
    cv2.destroyAllWindows()


def _0102():
    image_file = path_data + 'lena.jpg'
    img = cv2.imread(image_file)  # cv2.imread(image_file, cv2.IMREAD_COLOR)
    cv2.imwrite(path_data + 'lena.bmp', img)
    cv2.imwrite(path_data + 'lena.png', img)
    cv2.imwrite(path_data + 'lena2.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(path_data + 'lena2.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def _0103():
    image_file = path_data + 'lena.jpg'
    img_bgr = cv2.imread(image_file)  # cv2.IMREAD_COLOR
    plt.axis('off')
    # plt.imshow(imgBGR)
    # plt.show()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()


def _0104():
    img_bgr1 = cv2.imread(path_data + 'lena.jpg')
    img_bgr2 = cv2.imread(path_data + 'apple.jpg')
    img_bgr3 = cv2.imread(path_data + 'baboon.jpg')
    img_bgr4 = cv2.imread(path_data + 'orange.jpg')

    # 컬러 변환: BGR -> RGB
    img_rgb1 = cv2.cvtColor(img_bgr1, cv2.COLOR_BGR2RGB)
    img_rgb2 = cv2.cvtColor(img_bgr2, cv2.COLOR_BGR2RGB)
    img_rgb3 = cv2.cvtColor(img_bgr3, cv2.COLOR_BGR2RGB)
    img_rgb4 = cv2.cvtColor(img_bgr4, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
    # fig.canvas.set_window_title('Sample Pictures')

    ax[0][0].axis('off')
    ax[0][0].imshow(img_rgb1, aspect='auto')

    ax[0][1].axis('off')
    ax[0][1].imshow(img_rgb2, aspect='auto')

    ax[1][0].axis("off")
    ax[1][0].imshow(img_rgb3, aspect="auto")

    ax[1][0].axis("off")
    ax[1][1].imshow(img_rgb4, aspect='auto')

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0.05)
    plt.savefig(path_data + "0206.png", bbox_inches='tight')
    plt.show()


def _0105():
    cap = cv2.VideoCapture(0)  # 0번 카메라
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # cap = cv2.VideoCapture(path_data + 'vtest.avi')

    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('frame_size =', frame_size)

    while True:
        retval, frame = cap.read()  # 프레임 캡처
        if not retval:
            break

        cv2.imshow('frame', frame)

        key = cv2.waitKey(100)
        if key == 27:  # Esc
            break

    if cap.isOpened():
        cap.release()

    cv2.destroyAllWindows()


def _0106():
    url = 'https://www.youtube.com/watch?v=u_Q7Dkl7AIk'
    video = pafy.new(url)
    best = video.getbest(preftype='mp4')  # 'mp4','3gp’

    cap = cv2.VideoCapture(best.url)

    cnt = 0

    while True:
        cnt += 1
        retval, frame = cap.read()
        if not retval:
            break
        cv2.imshow('frame', frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        cv2.imshow('edges', edges)

        key = cv2.waitKey(25)
        if key == 27:  # Esc
            break

    cv2.destroyAllWindows()

    print(cnt)


def _0201():
    # White 배경 생성
    img = np.zeros(shape=(512, 512, 3), dtype=np.uint8) + 255
    # img = np.ones((512,512,3), np.uint8) * 255
    # img = np.full((512,512,3), (255, 255, 255), dtype= np.uint8)
    # img = np.zeros((512,512, 3), np.uint8) # Black 배경
    pt1 = 100, 100
    pt2 = 400, 400
    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)

    cv2.line(img, (0, 0), (500, 0), (255, 0, 0), 5)
    cv2.line(img, (0, 0), (0, 500), (0, 0, 255), 5)

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _0201()
