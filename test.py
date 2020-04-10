from keras.models import load_model as k_load_model
import numpy as np
import os
import cv2
import pandas as pd
import efficientnet.tfkeras
from tensorflow.python.keras.models import load_model
import platform
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


def circle_crop_v2(img):
    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))
    height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    return img


def ben_color2(image_path, sigmaX=10, scale=270, desired_size=300):
    lab = cv2.cvtColor(image_path, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    x = image[image.shape[0] // 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() // 2
    s = scale * 1.0 / r
    image = cv2.resize(image, (0, 0), fx=s, fy=s)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    image = cv2.resize(image, (desired_size, desired_size))
    image = circle_crop_v2(image)
    return image


def preprocess_image(image_path, desired_size=300):
    image = ben_color2(image_path, sigmaX=10, desired_size=desired_size)
    return image


def imgPrep(img, re_size=(300, 300), sigmaX=10, scale=270, crop=True, prepocess=True, mask=False):
    if crop:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        center_color = binary[int(binary.shape[0] / 2), int(binary.shape[1] / 2)]
        c_x, c_y = int(img.shape[0] / 2), int(img.shape[1] / 2)
        index_x = np.argwhere(binary[:, c_y] == 255) if center_color > 128 \
            else np.argwhere(binary[:, c_y] == 0)
        half_x = int(len(index_x) / 2)
        index_y = np.argwhere(binary[c_x, :] == 255) if center_color > 128 \
            else np.argwhere(binary[c_x, :] == 0)
        half_y = int(len(index_y) / 2)
        img_c = img[np.abs(c_x - half_x):c_x + half_x, np.abs(c_y - half_y):c_y + half_y]
    else:
        img_c = img
    if prepocess:
        lab = cv2.cvtColor(img_c, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        x = image[image.shape[0] // 2, :, :].sum(1)
        r = (x > x.mean() / 10).sum() // 2
        s = scale * 1.0 / r
        image = cv2.resize(image, (0, 0), fx=s, fy=s)

        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX),
                                -4, 128)
    else:
        image = img_c

    if mask:
        img_min = min(image.shape[0], image.shape[1])
        image = cv2.resize(image, (img_min, img_min))
        circle_img = np.zeros((img_min, img_min), np.uint8)
        cv2.circle(circle_img, (int(img_min / 2), int(img_min / 2)), int(img_min / 2), 1, thickness=-1)
        img_f = cv2.bitwise_and(image, image, mask=circle_img)
        return cv2.resize(img_f, re_size)
    else:
        return cv2.resize(image, re_size)


def tta(image, model, model_output='regression'):
    datagen = ImageDataGenerator()
    all_images = np.expand_dims(image, 0)
    hori_image = np.expand_dims(datagen.apply_transform(x=image, transform_parameters={"flip_horizontal": True}),
                                axis=0)
    vert_image = np.expand_dims(datagen.apply_transform(x=image, transform_parameters={"flip_vertical": True}), axis=0)
    rotated_image = np.expand_dims(datagen.apply_transform(x=image, transform_parameters={"theta": 15}), axis=0)
    all_images = np.append(all_images, hori_image, axis=0)
    all_images = np.append(all_images, vert_image, axis=0)
    all_images = np.append(all_images, rotated_image, axis=0)
    prediction = model.predict(all_images)
    if model_output is 'regression':
        return np.mean(prediction)
    else:
        prediction = np.sum(prediction, axis=0)
        return np.argmax(prediction)


def findAllFiles(root, file_type='jpg'):
    files = []

    def findOnelayer(base):
        for root, ds, fs in os.walk(base):
            ds.sort()
            fs.sort()
            for f in fs:
                if f.endswith('.' + file_type):
                    fullname = os.path.join(root, f)
                    yield fullname

    for i in findOnelayer(root):
        files.append(i)
    return files


def model_test(model_path=None,
                model_type='effnet',
                re_size=300,
                tta_flag=True,
                model_output='regression',
                fs=None):
    K.clear_session()
    if model_type is 'effnet':
        model = load_model(model_path)
    else:
        model = k_load_model(model_path)
    result = []
    for i, p in enumerate(fs):
        im_path = p
        im_path = '/'.join(im_path.split('\\')) if platform.system() is not 'Windows' else im_path
        img = cv2.imread(im_path)
        if model_type is 'effnet':
            img = preprocess_image(img, re_size)
        else:
            img = imgPrep(img, re_size=(re_size, re_size), mask=False)
        if tta_flag:
            out = tta(img, model, model_output)
            out = float(out)
            print(out, ' ', os.path.split(im_path)[-1])
            result.append(out)
        else:
            if model_output is 'regression':
                out = model.predict(img.reshape(1, re_size, re_size, 3))
                out = float(out)
                print(out, ' ', os.path.split(im_path)[-1])
                result.append(out)
            else:
                out = model.predict(img.reshape(1, re_size, re_size, 3))
                out = np.argmax(out)
                print(out, ' ', os.path.split(im_path)[-1])
                result.append(out)
    K.clear_session()
    return result


if __name__ == "__main__":
    data_root = r'./data/test'
    fs = findAllFiles(data_root, 'jpg')

    models = {
        'effnet_1_cl_tta': ['./all_models/EfficientNetB5-1.h5', 'effnet', 300, True, 'classify'],
        'effnet_2_cl_tta': ['./all_models/EfficientNetB5-2.h5', 'effnet', 300, True, 'classify'],
        'effnet_1_rg_tta': ['./all_models/EfficientNetB5-3.h5', 'effnet', 300, True, 'regression'],
        'effnet_2_rg_tta': ['./all_models/EfficientNetB5-4.h5', 'effnet', 300, True, 'regression'],
        'inpnet_1_rg': ['./all_models/InceptionResNetV2-1.h5', 'inpnet', 300, False, 'regression'],
        'inpnet_2_rg_tta': ['./all_models/InceptionResNetV2-2.h5', 'inpnet', 300, True, 'regression'], }
    result = {}
    for k, v in models.items():
        print(k)
        outs = model_test(v[0], v[1], v[2], v[3], v[4], fs=fs)
        result[k] = outs
    y_all = np.zeros((400, len(result)))
    for i, (k, v) in enumerate(result.items()):
        y_all[:, i] = np.array(v)

    y_pred = y_all.mean(axis=1)
    thrs = [0.5, 1.5, 2.45, 3.5]
    y_pred[y_pred < thrs[0]] = 0
    y_pred[(y_pred >= thrs[0]) & (y_pred < thrs[1])] = 1
    y_pred[(y_pred >= thrs[1]) & (y_pred < thrs[2])] = 2
    y_pred[(y_pred >= thrs[2]) & (y_pred < thrs[3])] = 3
    y_pred[y_pred >= thrs[3]] = 4
    y_pred = y_pred.astype('int')

    csv_path = './upload'
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    save_csv_path = os.path.join(csv_path,'Challenge1_upload.csv')
    image_id = []
    for i in fs:
        _,name = os.path.split(i)
        image_id.append(name.split('.')[0])
    df = pd.DataFrame(zip(image_id,y_pred), columns=['image_id', 'DR_Level'])
    df.to_csv(save_csv_path, index=False)


