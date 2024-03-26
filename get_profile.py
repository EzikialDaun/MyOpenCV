from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import urllib.request
import os

if __name__ == "__main__":
    searchKey = input('검색할 키워드 입력 : ')
    limit = int(input('이미지 수: '))
    path_directory = input('디렉토리 경로: ')


    def create_dir(path_dir):
        try:
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
        except OSError:
            print('Error')


    create_dir(path_directory)

    driver = webdriver.Chrome()
    driver.get('https://www.google.co.kr/imghp')

    # 쿼리 검색 및 검색 버튼 클릭
    elem = driver.find_element('name', 'q')
    elem.send_keys(searchKey)
    elem.send_keys(Keys.RETURN)

    # 안정적인 수집을 위한 딜레이
    time.sleep(20)
    images = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")  # 각 이미지들의 class
    for i, image in enumerate(images):
        try:
            if i > limit:
                break
            image.click()
            # 안정적인 수집을 위한 딜레이
            time.sleep(5)
            # 이 DOM 구조는 바뀔 수 있음
            # gif와 같은 형식은 예외 발생
            imgUrl = driver.find_element(By.XPATH,
                                         '//*[@id="Sva75c"]/div[2]/div[2]/div[2]/div[2]/c-wiz/div/div/div/div/div[3]/div[1]/a/img[1]').get_attribute(
                "src")
            imgUrl = imgUrl.replace('https', 'http')  # https로 요청할 경우 보안 문제로 SSL에러가 남
            opener = urllib.request.build_opener()
            opener.addheaders = [
                ('User-Agent', 'Mozilla/5.0')]  # https://docs.python.org/3/library/urllib.request.html 참고
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(imgUrl, f'{path_directory}/{str(i + 1)}.jpg')  # url을
        except Exception as e:
            print('Error : ', e)
            pass

    driver.close()
