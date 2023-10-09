import cv2
from deepface import DeepFace

if __name__ == "__main__":
    path_data = './data/'
    img_name = 'lena.jpg'
    objs = DeepFace.analyze(img_path=path_data + img_name,
                            actions=('age', 'gender', 'race', 'emotion'),
                            )
    img = cv2.imread(path_data + img_name)

    for obj in objs:
        key_region = 'region'
        point_x, point_y, width, height = obj[key_region]['x'], obj[key_region]['y'], obj[key_region]['w'], \
                                          obj[key_region]['h']
        cv2.rectangle(img, (point_x, point_y), (point_x + width, point_y + height), (0, 0, 255), 2)
        print(obj)

    cv2.imshow('image', img)

    cv2.waitKey()
    cv2.destroyAllWindows()
