import cv2
from matplotlib import pyplot as plt
import numpy as np
import pafy
import time

path_data = './data/'


# 사진 파일 읽기
def _0101():
    image_file = path_data + 'lena.jpg'
    img = cv2.imread(image_file)  # cv2.IMREAD_COLOR
    img2 = cv2.imread(image_file, 0)  # cv2.IMREAD_GRAYSCALE
    cv2.imshow('Lena color', img)
    cv2.imshow('Lena grayscale', img2)

    cv2.waitKey()
    cv2.destroyAllWindows()


# 사진 파일 쓰기
def _0102():
    image_file = path_data + 'lena.jpg'
    img = cv2.imread(image_file)  # cv2.imread(image_file, cv2.IMREAD_COLOR)
    cv2.imwrite(path_data + 'lena.bmp', img)
    cv2.imwrite(path_data + 'lena.png', img)
    cv2.imwrite(path_data + 'lena2.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(path_data + 'lena2.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])


# pyplot으로 사진 읽기
def _0103():
    image_file = path_data + 'lena.jpg'
    img_bgr = cv2.imread(image_file)  # cv2.IMREAD_COLOR
    plt.axis('off')
    # plt.imshow(imgBGR)
    # plt.show()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()


# 사진 4개 서브플롯으로 띄우기
# OpenCV는 BGR 체계를,
# pyplot은 RGB 체계를 사용해서 변환 필요
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


# 비디오 캡쳐
# 캠이나 로컬 영상 파일을 캡쳐
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


# 이미지에 사각형과 선분 그리기
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


# 사각형을 관통하는 선분의 출발점과 도착점 추출
def _0202():
    img = np.zeros(shape=(512, 512, 3), dtype=np.uint8) + 255

    x1, x2 = 100, 400
    y1, y2 = 100, 400
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))

    pt1 = 340, 50
    pt2 = 300, 500
    cv2.line(img, pt1, pt2, (255, 0, 0), 2)

    pt3 = 320, 50
    pt4 = 200, 450
    cv2.line(img, pt3, pt4, (255, 255, 0), 2)

    img_rect = (x1, y1, x2 - x1, y2 - y1)
    retval, rpt1, rpt2 = cv2.clipLine(img_rect, pt1, pt2)
    if retval:
        cv2.circle(img, rpt1, radius=5, color=(0, 255, 0), thickness=-1)
        cv2.circle(img, rpt2, radius=5, color=(0, 255, 0), thickness=-1)

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def _0202_ex():
    img = np.zeros(shape=(512, 512, 3), dtype=np.uint8) + 255

    rect_x1, rect_y1 = 50, 50
    rect_x2, rect_y2 = 250, 250

    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0))
    cv2.circle(img, (150, 150), radius=100, color=(0, 0, 0), thickness=2)
    cv2.line(img, (200, 25), (75, 400), (0, 0, 0), 2)

    # clipLine에 인수로 들어가는 사각형은 (시작 x 좌표, 시작 y 좌표, x 길이, y 길이)의 튜플 형태
    retval, rpt1, rpt2 = cv2.clipLine((rect_x1, rect_y1, rect_x2 - rect_x1, rect_y2 - rect_y1), (200, 25), (75, 400))
    if retval:
        cv2.circle(img, rpt1, radius=5, color=(0, 255, 0), thickness=-1)
        cv2.circle(img, rpt2, radius=5, color=(0, 255, 0), thickness=-1)

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 타원 그리기
def _0203():
    img = np.zeros(shape=(512, 512, 3), dtype=np.uint8) + 255
    pt_center = img.shape[0] // 2, img.shape[1] // 2
    size = 200, 100

    cv2.ellipse(img, pt_center, size, 0, 0, 360, (255, 0, 0))
    cv2.ellipse(img, pt_center, size, 45, 0, 360, (0, 0, 255))

    box = (pt_center, size, 0)
    cv2.ellipse(img, box, (255, 0, 0), 5)

    box = (pt_center, size, 45)
    cv2.ellipse(img, box, (0, 0, 255), 5)

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 임의의 포인트를 이어서 다각형 만들기
def _0204():
    img = np.zeros(shape=(512, 512, 3), dtype=np.uint8) + 255

    pts1 = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
    pts2 = np.array([[300, 200], [400, 100], [400, 200]])

    cv2.polylines(img, [pts1, pts2], isClosed=True, color=(255, 0, 0))

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 타원에서 포인트를 추출하여 타원과 유사한 다각형 만들기
def _0205():
    img = np.zeros(shape=(512, 512, 3), dtype=np.uint8) + 255

    pt_center = img.shape[0] // 2, img.shape[1] // 2
    size = 200, 100

    cv2.ellipse(img, pt_center, size, 0, 0, 360, (255, 0, 0))
    pts1 = cv2.ellipse2Poly(pt_center, size, 0, 0, 360, delta=45)

    cv2.ellipse(img, pt_center, size, 45, 0, 360, (255, 0, 0))
    pts2 = cv2.ellipse2Poly(pt_center, size, 45, 0, 360, delta=45)

    cv2.polylines(img, [pts1, pts2], isClosed=True, color=(0, 0, 255))

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 텍스트 띄우기
def _0206():
    img = np.zeros(shape=(512, 512, 3), dtype=np.uint8) + 255
    text = 'OpenCV Programming'
    org = (50, 100)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, org, font, 1, (255, 0, 0), 2)

    size, base_line = cv2.getTextSize(text, font, 1, 2)
    # print('size=', size)
    # print('base_line=', base_line)
    cv2.rectangle(img, org, (org[0] + size[0], org[1] - size[1]), (0, 0, 255))
    cv2.circle(img, org, 3, (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 키보드 입력
def _0207():
    width, height = 512, 512
    x, y, radius = 256, 256, 50
    direction = 0  # right

    while True:
        # 인수는 대기 시간
        key = cv2.waitKeyEx(30)

        if key == 0x1B:
            break

        # 방향키 방향전환
        elif key == 0x270000:  # right
            direction = 0
        elif key == 0x280000:  # down
            direction = 1
        elif key == 0x250000:  # left
            direction = 2
        elif key == 0x260000:  # up
            direction = 3

        # 방향으로 이동
        if direction == 0:  # right
            x += 10
        elif direction == 1:  # down
            y += 10
        elif direction == 2:  # left
            x -= 10
        else:  # 3, up
            y -= 10

        # 경계확인
        if x < radius:
            x = radius
            direction = 0
        if x > width - radius:
            x = width - radius
            direction = 2
        if y < radius:
            y = radius
            direction = 1
        if y > height - radius:
            y = height - radius
            direction = 3

        # 지우고, 그리기
        img = np.zeros((width, height, 3), np.uint8) + 255  # 지우기
        cv2.circle(img, (x, y), radius, (0, 0, 255), -1)
        cv2.imshow('img', img)

    cv2.destroyAllWindows()


def _0207_ex():
    width, height = 512, 512
    origin = (width / 2, height / 2)
    # mode 0 : 15도 회전
    # mode 1 : 30도 회전
    # mode 2 : -15도 회전
    # mode 3 : -30도 회전
    mode = 0
    angle = 0

    while True:
        # 인수는 대기 시간
        key = cv2.waitKeyEx(100)

        if key == 0x1B:
            break

        # 방향키 모드 전환
        elif key == 0x270000:  # right
            mode = 0
        elif key == 0x280000:  # down
            mode = 1
        elif key == 0x250000:  # left
            mode = 2
        elif key == 0x260000:  # up
            mode = 3

        # 모드에 따라 각도 회전
        if mode == 0:
            angle += 15
        elif mode == 1:
            angle += 30
        elif mode == 2:
            angle -= 15
        else:
            angle -= 30

        # 지우고, 그리기
        img = np.zeros((width, height, 3), np.uint8) + 255  # 지우기
        cv2.ellipse(img, (origin, (200, 100), angle), (0, 0, 255), 2)
        cv2.imshow('img', img)

    cv2.destroyAllWindows()


# 마우스 입력
def _0208():
    def on_mouse(event, x, y, flags, param):
        #    global img
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                cv2.rectangle(param[0], (x - 5, y - 5), (x + 5, y + 5), (255, 0, 0))
            else:
                cv2.circle(param[0], (x, y), 5, (255, 0, 0), 3)
        elif event == cv2.EVENT_LBUTTONUP:
            cv2.circle(param[0], (x, y), 5, (255, 0, 0), 3)
        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(param[0], (x, y), 5, (0, 0, 255), 3)
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            param[0] = np.zeros(param[0].shape, np.uint8) + 255
        elif event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                cv2.circle(param[0], (x, y), 3, (255, 0, 0), -1)
        cv2.imshow("img", param[0])

    img = np.zeros((512, 512, 3), np.uint8) + 255
    cv2.imshow('img', img)
    cv2.setMouseCallback('img', on_mouse, [img])
    cv2.waitKey()
    cv2.destroyAllWindows()


# 트랙바 컨트롤
def _0209():
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.imshow('img', img)

    def on_change(pos):  # 트랙바 핸들러
        r = cv2.getTrackbarPos('R', 'img')
        g = cv2.getTrackbarPos('G', 'img')
        b = cv2.getTrackbarPos('B', 'img')
        img[:] = (b, g, r)
        cv2.imshow('img', img)

    # 트랙바 생성
    cv2.createTrackbar('R', 'img', 0, 255, on_change)
    cv2.createTrackbar('G', 'img', 0, 255, on_change)
    cv2.createTrackbar('B', 'img', 0, 255, on_change)

    # 트랙바 위치 초기화
    # cv2.setTrackbarPos('R', 'img', 0)
    # cv2.setTrackbarPos('G', 'img', 0)
    cv2.setTrackbarPos('B', 'img', 255)

    cv2.waitKey()
    cv2.destroyAllWindows()


# 이미지 정보 추출
def _0210():
    img = cv2.imread(path_data + 'lena.jpg')  # cv2.IMREAD_COLOR
    # img = cv2.imread('./data/lena.jpg', cv2.IMREAD_GRAYSCALE)

    print('img.ndim=', img.ndim)
    print('img.shape=', img.shape)
    print('img.dtype=', img.dtype)

    # np.bool, np.uint16, np.uint32, np.float32, np.float64, np.complex64
    img = img.astype(np.int32)
    print('img.dtype=', img.dtype)

    img = np.uint8(img)
    print('img.dtype=', img.dtype)


# 넘파이 배열 차원 변환
def _0211():
    img = cv2.imread(path_data + 'lena.jpg', cv2.IMREAD_GRAYSCALE)
    print('img.shape=', img.shape)

    # img = img.reshape(img.shape[0]*img.shape[1])
    img = img.flatten()
    print('img.shape=', img.shape)

    img = img.reshape(-1, 512, 512)
    print('img.shape=', img.shape)

    cv2.imshow('img', img[0])
    cv2.waitKey()
    cv2.destroyAllWindows()


# 트랙바와 마우스 입력을 받아 그림판 표현
def _0211_ex():
    img = cv2.imread(path_data + 'lena.jpg')
    title = 'lena'
    r, g, b, thickness = 0, 0, 0, 0

    def on_r_change(pos):  # 트랙바 핸들러
        nonlocal r
        r = pos

    def on_g_change(pos):  # 트랙바 핸들러
        nonlocal g
        g = pos

    def on_b_change(pos):  # 트랙바 핸들러
        nonlocal b
        b = pos

    def on_t_change(pos):  # 트랙바 핸들러
        nonlocal thickness
        thickness = pos

    def on_mouse(event, x, y, flags, param):
        nonlocal r, g, b
        if event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                cv2.circle(param[0], (x, y), thickness, (b, g, r), -1)
                cv2.imshow(title, param[0])

    cv2.imshow(title, img)
    cv2.setMouseCallback(title, on_mouse, [img])

    # 트랙바 생성
    cv2.createTrackbar('R', title, 0, 255, on_r_change)
    cv2.createTrackbar('G', title, 0, 255, on_g_change)
    cv2.createTrackbar('B', title, 0, 255, on_b_change)
    cv2.createTrackbar('thickness', title, 0, 20, on_t_change)
    cv2.setTrackbarPos('thickness', title, 3)

    cv2.waitKey()
    cv2.destroyAllWindows()


# ROI 접근하여 영역 수정
def _0212():
    img = cv2.imread(path_data + 'lena.jpg', cv2.IMREAD_GRAYSCALE)
    img[100, 200] = 0  # 화소값(밝기,그레이스케일) 변경
    print(img[100:110, 200:210])  # ROI 접근

    # for y in range(100, 400):
    #    for x in range(200, 300):
    #        img[y, x] = 0

    img[100:400, 200:300] = 0  # ROI 접근

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 컬러 이미지 ROI 접근
def _0213():
    img = cv2.imread(path_data + 'lena.jpg')  # cv2.IMREAD_COLOR
    img[100, 200] = [255, 0, 0]  # 컬러(BGR) 변경
    print(img[100, 200:210])  # ROI 접근

    # for y in range(100, 400):
    #     for x in range(200, 300):
    #         img[y, x] = [255, 0, 0]    # 파랑색(blue)으로 변경

    img[100:400, 200:300] = [255, 0, 0]  # ROI 접근

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 컬러 이미지 채널별 접근
def _0214():
    img = cv2.imread(path_data + 'lena.jpg')  # cv2.IMREAD_COLOR

    # for y in range(100, 400):
    #     for x in range(200, 300):
    #         img[y, x, 0] = 255      # B-채널을 255로 변경

    img[100:400, 200:300, 0] = 255  # B-채널을 255로 변경
    img[100:400, 300:400, 1] = 255  # G-채널을 255로 변경
    img[100:400, 400:500, 2] = 255  # R-채널을 255로 변경

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# ROI를 위한 인덱싱 연습
def _0214_ex():
    img = cv2.imread(path_data + 'lena.jpg')  # cv2.IMREAD_COLOR

    img[10:110, 200:400, 2] = 255
    img[110:210, 200:400, 1] = 255
    img[210:310, 200:400, 0] = 255
    img[310:410, 200:400] = [0, 0, 0]
    img[410:510, 200:400] = [255, 255, 255]

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# add 메서드로 밝기 조절
def _0301():
    src1 = cv2.imread(path_data + 'lena.jpg', cv2.IMREAD_GRAYSCALE)
    src2 = np.zeros(shape=(512, 512), dtype=np.uint8) + 100

    dst1 = src1 + src2
    dst2 = cv2.add(src1, src2)
    # dst2 = cv2.add(src1, src2, dtype = cv2.CV_8U)

    cv2.imshow('dst1', dst1)
    cv2.imshow('dst2', dst2)
    cv2.waitKey()
    cv2.destroyAllWindows()


def _0301_ex():
    src1 = cv2.imread(path_data + 'lena.jpg')
    src2 = np.zeros(shape=(512, 512, 3), dtype=np.uint8) + 100

    dst1 = src1 + src2
    dst2 = cv2.add(src1, src2)
    # dst2 = cv2.add(src1, src2, dtype = cv2.CV_8U)

    cv2.imshow('src1', src1)
    cv2.imshow('dst1', dst1)
    cv2.imshow('dst2', dst2)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 비트 연산으로 이미지 마스킹
def _0302():
    img_file1 = path_data + 'lena.jpg'
    img_file2 = path_data + 'orange.jpg'
    # image read
    img1 = cv2.imread(img_file1)
    img2 = cv2.imread(img_file2)

    bit_and = cv2.bitwise_and(img1, img2)
    bit_or = cv2.bitwise_or(img2, img1)
    bit_not = cv2.bitwise_not(img2)
    bit_xor = cv2.bitwise_xor(img2, img1)
    cv2.imshow("bit_and", bit_and)
    cv2.imshow("bit_or", bit_or)
    cv2.imshow("bit_not", bit_not)
    cv2.imshow("bit_xor", bit_xor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 히스토그램 정규화
# 특정 부분에 몰린 값을 골고루 분포하게 만듬
# 형태 유지, 빽빽하게
def _0303():
    src = cv2.imread(path_data + 'lena.jpg', cv2.IMREAD_GRAYSCALE)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(src)
    print('src:', min_val, max_val, min_loc, max_loc)

    dst = cv2.normalize(src, None, 100, 200, cv2.NORM_MINMAX)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dst)
    print('dst:', min_val, max_val, min_loc, max_loc)

    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 임계값 컨트롤
def _0304():
    src = cv2.imread(path_data + 'lena.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('src', src)

    ret, dst = cv2.threshold(src, 120, 255, cv2.THRESH_BINARY)
    print('ret=', ret)
    cv2.imshow('dst', dst)

    ret2, dst2 = cv2.threshold(src, 200, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print('ret2=', ret2)
    cv2.imshow('dst2', dst2)

    cv2.waitKey()
    cv2.destroyAllWindows()


def _0305():
    src = cv2.imread(path_data + 'lena.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('src', src)

    ret, dst = cv2.threshold(src, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('dst', dst)

    dst2 = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 51, 7)
    cv2.imshow('dst2', dst2)

    dst3 = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 51, 7)
    cv2.imshow('dst3', dst3)

    cv2.waitKey()
    cv2.destroyAllWindows()


# 히스토그램 계산
# 일정 값이 얼마나 존재하는지
def _0306():
    src = np.array([[0, 0, 0, 0],
                    [1, 1, 3, 5],
                    [6, 1, 1, 3],
                    [4, 3, 1, 7]
                    ], dtype=np.uint8)

    hist1 = cv2.calcHist(images=[src], channels=[0], mask=None,
                         histSize=[8], ranges=[0, 8])
    print('hist1 = ', hist1)

    hist2 = cv2.calcHist(images=[src], channels=[0], mask=None,
                         histSize=[4], ranges=[0, 4])
    print('hist2 = ', hist2)

    hist1 = hist1.flatten()
    hist2 = hist2.flatten()

    plt.title('hist1: histSize = 8, range 0 ~ 8')
    bin_x = np.arange(8)
    plt.bar(bin_x, hist1, width=1, color='b')
    plt.show()

    plt.title('hist2: histSize = 4, range 0 ~ 4')
    bin_x = np.arange(4)
    plt.bar(bin_x, hist2, width=1, color='b')
    plt.show()


def _0307():
    src = cv2.imread(path_data + 'lena.jpg', cv2.IMREAD_GRAYSCALE)

    hist1 = cv2.calcHist(images=[src], channels=[0], mask=None,
                         histSize=[32], ranges=[0, 256])

    hist2 = cv2.calcHist(images=[src], channels=[0], mask=None,
                         histSize=[256], ranges=[0, 256])

    # 다차원 배열을 1차원 배열로 직렬화
    hist1 = hist1.flatten()
    hist2 = hist2.flatten()

    # 2
    plt.title('hist1: binX = np.arange(32)')
    bin_x = np.arange(32)
    plt.bar(bin_x, hist1, width=1, color='b')
    plt.show()

    # 4
    plt.title('hist2: binX = np.arange(256)')
    bin_x = np.arange(256)
    plt.bar(bin_x, hist2, width=1, color='b')
    plt.show()


# 컬러 이미지 채널별 히스토그램 계산
def _0308():
    src = cv2.imread(path_data + 'lena.jpg')
    hist_color = ('b', 'g', 'r')
    for i in range(3):
        hist = cv2.calcHist(images=[src], channels=[i], mask=None,
                            histSize=[256], ranges=[0, 256])
        plt.plot(hist, color=hist_color[i])
    plt.show()


# 그레이스케일 이미지 히스토그램 계산 함수 직접 만드는 연습
def _0308_ex():
    src = cv2.imread(path_data + 'lena.jpg', cv2.IMREAD_GRAYSCALE)

    src_hist = cv2.calcHist(images=[src], channels=[0], mask=None,
                            histSize=[256], ranges=[0, 256])

    src_hist = src_hist.flatten()

    def draw_gray_hist(image):
        temp = image.flatten()
        result = np.zeros(256)
        for i in range(256):
            result[i] = np.sum(temp == i)
        return result

    my_hist = draw_gray_hist(src)

    plt.title('src_hist')
    bin_x = np.arange(256)
    plt.bar(bin_x, src_hist, width=1, color='b')
    plt.show()

    plt.title('my_hist')
    bin_x = np.arange(256)
    plt.bar(bin_x, my_hist, width=1, color='b')
    plt.show()


# 히스토그램 평활화
# 어둡거나 밝은 영상을 선명하게 해줌
def _0309():
    src = np.array([[0, 0, 0, 0],
                    [1, 1, 3, 5],
                    [6, 1, 1, 3],
                    [4, 3, 1, 7]], dtype=np.uint8)

    dst = cv2.equalizeHist(src)
    print('dst =', dst)


def _0310():
    src = cv2.imread(path_data + 'lena.jpg', cv2.IMREAD_GRAYSCALE)
    dst = cv2.equalizeHist(src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

    plt.title('Grayscale histogram of lena.jpg')

    hist1 = cv2.calcHist(images=[src], channels=[0], mask=None,
                         histSize=[256], ranges=[0, 256])
    plt.plot(hist1, color='b', label='hist1 in src')

    hist2 = cv2.calcHist(images=[dst], channels=[0], mask=None,
                         histSize=[256], ranges=[0, 256])
    plt.plot(hist2, color='r', alpha=0.7, label='hist2 in dst')
    plt.legend(loc='best')
    plt.show()


def _0310_ex():
    src = cv2.imread(path_data + 'target.jpg', cv2.IMREAD_GRAYSCALE)
    dst = cv2.equalizeHist(src)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

    cv2.imwrite(path_data + 'result.jpg', dst)


# 히스토그램 역투영
# 비슷한 색 기준으로 영역 분할
def _0311():
    src = np.array([[0, 0, 0, 0],
                    [1, 1, 3, 5],
                    [6, 1, 1, 3],
                    [4, 3, 1, 7]
                    ], dtype=np.uint8)

    hist = cv2.calcHist(images=[src], channels=[0], mask=None,
                        histSize=[4], ranges=[0, 8])
    print('hist = ', hist)

    back_p = cv2.calcBackProject([src], [0], hist, [0, 8], scale=1)
    print('backP = ', back_p)


# selectROI로 관심 영역 마우스 드래그로 선택
# 선택된 ROI와 유사한 색상 강조
def _0312():
    # 1
    src = cv2.imread(path_data + 'lena.jpg')
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # 2
    roi = cv2.selectROI(src)
    print('roi =', roi)
    roi_h = h[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    hist = cv2.calcHist([roi_h], [0], None, [64], [0, 256])
    back_p = cv2.calcBackProject([h.astype(np.float32)], [0], hist, [0, 256], scale=1.0)
    # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(backP)
    # T = maxVal -1 # threshold
    # 3
    hist = cv2.sort(hist, cv2.SORT_EVERY_COLUMN + cv2.SORT_DESCENDING)
    k = 1
    T = hist[k][0] - 1  # threshold
    print('T =', T)
    ret, dst = cv2.threshold(back_p, T, 255, cv2.THRESH_BINARY)

    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 히스토그램 유사도 비교
def _0313():
    # 1
    n_points = 100000
    pts1 = np.zeros((n_points, 1), dtype=np.uint16)
    pts2 = np.zeros((n_points, 1), dtype=np.uint16)

    cv2.setRNGSeed(int(time.time()))
    cv2.randn(pts1, mean=128, stddev=10)
    cv2.randn(pts2, mean=110, stddev=20)

    # 2
    h1 = cv2.calcHist(images=[pts1], channels=[0], mask=None,
                      histSize=[256], ranges=[0, 256])
    cv2.normalize(h1, h1, 1, 0, cv2.NORM_L1)
    plt.plot(h1, color='r', label='H1')
    h2 = cv2.calcHist(images=[pts2], channels=[0], mask=None,
                      histSize=[256], ranges=[0, 256])
    cv2.normalize(h2, h2, 1, 0, cv2.NORM_L1)

    # 3
    d1 = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    d2 = cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR)
    d3 = cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)
    d4 = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
    print('d1(H1, H2, CORREL) =', d1)
    print('d2(H1, H2, CHISQR)=', d2)
    print('d3(H1, H2, INTERSECT)=', d3)
    print('d4(H1, H2, BHATTACHARYYA)=', d4)

    plt.plot(h2, color='b', label='H2')
    plt.legend(loc='best')
    plt.show()


def _0314():
    # 1
    n_points = 100000
    pts1 = np.zeros((n_points, 1), dtype=np.uint16)
    pts2 = np.zeros((n_points, 1), dtype=np.uint16)

    cv2.setRNGSeed(int(time.time()))
    cv2.randn(pts1, mean=128, stddev=10)
    cv2.randn(pts2, mean=110, stddev=20)

    # 2
    H1 = cv2.calcHist(images=[pts1], channels=[0], mask=None,
                      histSize=[256], ranges=[0, 256])
    # cv2.normalize(H1, H1, norm_type=cv2.NORM_L1)

    H2 = cv2.calcHist(images=[pts2], channels=[0], mask=None,
                      histSize=[256], ranges=[0, 256])
    # cv2.normalize(H2, H2, norm_type=cv2.NORM_L1)

    # 3
    s1 = np.zeros((H1.shape[0], 2), dtype=np.float32)
    s2 = np.zeros((H1.shape[0], 2), dtype=np.float32)
    # s1[:, 0] = H1[:, 0]
    # s2[:, 0] = H2[:, 0]
    for i in range(s1.shape[0]):
        s1[i, 0] = H1[i, 0]
        s2[i, 0] = H2[i, 0]
        s1[i, 1] = i
        s2[i, 1] = i

    emd1, lower_bound, flow = cv2.EMD(s1, s2, cv2.DIST_L1)
    print('EMD(S1, S2, DIST_L1) =', emd1)

    emd2, lower_bound, flow = cv2.EMD(s1, s2, cv2.DIST_L2)
    print('EMD(S1, S2, DIST_L2) =', emd2)

    emd3, lower_bound, flow = cv2.EMD(s1, s2, cv2.DIST_C)
    print('EMD(S1, S2, DIST_C) =', emd3)

    plt.plot(H1, color='r', label='H1')
    plt.plot(H2, color='b', label='H2')
    plt.legend(loc='best')
    plt.show()


# src 사진과 가장 유사한 사진을 찾는 연습
def _0314_ex():
    cmp_image_names = ['cmp2.jpg', 'cmp1.jpg', 'cmp3.jpg', 'cmp4.jpg']

    src_gray = cv2.imread(path_data + 'cmp_org.jpg', cv2.IMREAD_GRAYSCALE)

    cmp_gray_images = []
    for name in cmp_image_names:
        cmp_gray_images.append(cv2.imread(path_data + name, cv2.IMREAD_GRAYSCALE))

    src_gray_hist = cv2.calcHist(images=[src_gray], channels=[0], mask=None,
                                 histSize=[256], ranges=[0, 256])

    cmp_gray_hists = []
    for image in cmp_gray_images:
        hist = cv2.calcHist(images=[image], channels=[0], mask=None,
                            histSize=[256], ranges=[0, 256])
        cmp_gray_hists.append(hist)

    cmp_result = []

    for hist in cmp_gray_hists:
        cmp_result.append(cv2.compareHist(src_gray_hist, hist, cv2.HISTCMP_BHATTACHARYYA))

    similar_image = cmp_image_names[cmp_result.index(min(cmp_result))]

    print(f'similar image is {similar_image}')


if __name__ == "__main__":
    _0314_ex()
