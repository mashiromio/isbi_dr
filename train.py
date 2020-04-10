from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from keras.callbacks import Callback, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.optimizers import Adam
from keras.models import Sequential
import os, time
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import cv2
from tqdm import tqdm
import platform
import efficientnet.keras as efn
import argparse


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


class kappa_call(Callback):
    def __init__(self,
                 filepath,
                 validation_data=None,
                 monitor='kappa',
                 tta_flag=True,
                 model_output='regression',
                 max_nums=1, mode='auto'):
        super(kappa_call, self).__init__()
        self.filepath = filepath
        self.x_val, self.y_val = validation_data
        self.monitor = monitor
        self.tta_flag = tta_flag
        self.model_output = model_output
        self.max_num = max_nums
        self.save_models = []
        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'loss' in self.monitor:
                self.monitor_op = np.less
                self.best = np.Inf
            else:
                self.monitor_op = np.greater
                self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred = np.empty(self.x_val.shape[0], dtype=np.float)
        for i, x in enumerate(self.x_val):
            if self.tta_flag:
                y = tta(x, self.model, self.model_output)
                y_pred[i] = y
            else:
                y = self.model.predict(x.reshape((1, x.shape[0], x.shape[1], x.shape[2])))
                if self.model_output is 'regression':
                    y_pred[i] = y.reshape(-1)
                else:
                    y_pred[i] = np.argmax(y, axis=1)
        if self.model_output is 'regression':
            thrs = [0.5, 1.5, 2.5, 3.5]
            y_pred[y_pred < thrs[0]] = 0
            y_pred[(y_pred >= thrs[0]) & (y_pred < thrs[1])] = 1
            y_pred[(y_pred >= thrs[1]) & (y_pred < thrs[2])] = 2
            y_pred[(y_pred >= thrs[2]) & (y_pred < thrs[3])] = 3
            y_pred[y_pred >= thrs[3]] = 4
            y_pred = y_pred.astype('int')
        else:
            y_pred = y_pred.astype('int')
        y_true = self.y_val if self.model_output is 'regression' else np.argmax(self.y_val, axis=1)
        valid_matrix = confusion_matrix(y_true, y_pred)
        print(valid_matrix)
        if self.monitor is 'kappa':
            val_kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
            print(val_kappa)
            logs['kappa'] = val_kappa
            current = val_kappa
        else:
            current = logs.get(self.monitor)

        if self.monitor_op(current, self.best):
            print('\nEpoch {:05d}: {} improved from {:0.5f} to {:0.5f},'
                  .format(epoch + 1, self.monitor, self.best, current, ))
            self.best = current
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if len(self.save_models) < self.max_num:
                self.save_models.append(filepath)
            else:
                os.remove(self.save_models.pop(0))
                self.save_models.append(filepath)
            self.model.save(filepath)
            print('\nsaving model to {}'.format(filepath))
        else:
            print('\nEpoch {:05d}: {} did not improve from {:0.5f}'
                  .format(epoch + 1, self.monitor, self.best))


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


def myDataload_isbi(data_root=r'./data/isbi', csv_path=None, re_size=(300, 300)):
    csv_path_u = '/'.join(csv_path.split('\\')) if platform.system() is not 'Windows' else csv_path
    csv_file = pd.read_csv(csv_path_u)
    df = pd.DataFrame(csv_file)
    true_labels = df.loc[:, ['left_eye_DR_Level', 'right_eye_DR_Level']].sum(axis=1)
    samples = len(true_labels)

    images = np.empty((samples, re_size[0], re_size[1], 3), dtype=np.uint8)
    lables = np.empty(samples, dtype=np.float)

    for i, p in enumerate(tqdm(df['image_path'])):
        im_path = data_root + p
        im_path = '/'.join(im_path.split('\\')) if platform.system() is not 'Windows' else im_path
        img = cv2.imread(im_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = imgPrep(img, re_size=re_size)
        img = img.reshape(1, re_size[0], re_size[1], 3)
        images[i, :, :, :] = img
        lables[i] = int(true_labels[i])

    return images, lables


def myDataload_aptos(data_root=r'./data/aptos', csv_path=None, re_size=(300, 300)):
    csv_path_u = '/'.join(csv_path.split('\\')) if platform.system() is not 'Windows' else csv_path
    csv_file = pd.read_csv(csv_path_u)
    df = pd.DataFrame(csv_file)
    true_labels = df['diagnosis'].values.tolist()
    samples = len(true_labels)
    images = np.empty((samples, re_size[0], re_size[1], 3), dtype=np.uint8)
    lables = np.empty(samples, dtype=np.float)

    for i, p in enumerate(tqdm(df['id_code'])):
        im_path = os.path.join(data_root, 'train_images', p + '.png')
        im_path = '/'.join(im_path.split('\\')) if platform.system() is not 'Windows' else im_path
        img = cv2.imread(im_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = imgPrep(img, re_size=re_size)
        img = img.reshape(1, re_size[0], re_size[1], 3)
        images[i, :, :, :] = img
        lables[i] = int(true_labels[i])

    return images, lables


def my_model(model_type='effnet',
             model_output='regression',
             re_size=(300, 300),
             weights=True):
    if model_type is 'effnet':
        top_model = efn.EfficientNetB5(weights=None,
                                       include_top=False,
                                       input_shape=(re_size[0], re_size[1], 3),
                                       pooling='avg')
        if weights:
            # download from https://github.com/qubvel/efficientnet/releases
            top_model.load_weights('efficientnet-b5_imagenet_1000_notop.h5')
    else:
        top_model = InceptionResNetV2(weights=None,
                                      include_top=False,
                                      input_shape=(re_size[0], re_size[1], 3),
                                      pooling='avg')
        if weights:
            # download from https://www.kaggle.com/keras/inceptionresnetv2
            top_model.load_weights('inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model = Sequential(name=top_model.name)
    model.add(top_model)
    model.add(Dropout(0.5))
    if model_output is 'regression':
        model.add(Dense(1, activation='linear', name='probs'))
    else:
        model.add(Dense(5, activation='softmax', name='probs'))
    return model


def main(args):
    model_type = args.model_type
    model_output = args.model_output
    re_size = (args.re_size, args.re_size)
    batch_size = args.batch_size
    epochs = args.epochs
    weights = args.weights
    tta_flag = args.tta_flag

    isbi_root = os.path.join(args.dataset_path, 'isbi')
    isbi_t_csv_path = os.path.join(isbi_root, r'regular-fundus-training/regular-fundus-training.csv')
    isbi_v_csv_path = os.path.join(isbi_root, r'regular-fundus-validation/regular-fundus-validation.csv')
    i_t_images, i_t_lables = myDataload_isbi(data_root=isbi_root, csv_path=isbi_t_csv_path,
                                             re_size=re_size)
    i_v_images, i_v_lables = myDataload_isbi(data_root=isbi_root, csv_path=isbi_v_csv_path,
                                             re_size=re_size)
    aptos_root = os.path.join(args.dataset_path, 'aptos')
    aptos_t_csv_path = os.path.join(aptos_root, 'train.csv')
    a_t_images, a_t_lables = myDataload_aptos(data_root=aptos_root, csv_path=aptos_t_csv_path,
                                              re_size=re_size)
    t_images = np.concatenate((i_t_images, a_t_images))
    t_lables = np.concatenate((i_t_lables, a_t_lables))
    v_images = i_v_images
    v_lables = i_v_lables

    if model_output is not 'regression':
        v_lables = to_categorical(v_lables)
        t_lables = to_categorical(t_lables)
    model = my_model(model_type=model_type,
                     model_output=model_output,
                     re_size=re_size,
                     weights=weights)
    model.summary()
    my_datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='constant',
        cval=0,
        horizontal_flip=True,
        vertical_flip=True, )
    data_generator = my_datagen.flow(t_images, t_lables, batch_size=batch_size, seed=88)
    opt = Adam(lr=0.0005)
    if model_output is 'regression':
        loss = 'mse'
    else:
        loss = 'categorical_crossentropy'
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['acc'], )
    monitor = 'kappa'
    st = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    mt = '-rg' if model_output is 'regression' else '-cl'
    tta_f = '-tta' if tta_flag else '-no_tta'
    model_dir = os.path.join(args.model_path, st + mt + tta_f)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, model.name + mt + tta_f +
                              '_{epoch:03d}_monitor-{monitor:.5f}.h5'
                              .replace('monitor', monitor))
    kappa_callback = kappa_call(model_path,
                                validation_data=(v_images, v_lables),
                                monitor=monitor,
                                model_output=model_output,
                                tta_flag=tta_flag)
    RL = ReduceLROnPlateau(monitor='acc', factor=0.5, verbose=1,
                           patience=5, min_lr=0.0001)
    callbacks = [kappa_callback, RL]
    model.fit_generator(
        data_generator,
        steps_per_epoch=t_images.shape[0] / batch_size,
        epochs=epochs,
        validation_data=(v_images, v_lables),
        validation_steps=v_images.shape[0] / batch_size,
        callbacks=callbacks,
        verbose=1, )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', type=str, default='./data', )
    parser.add_argument('-mp', '--model_path', type=str, default='./models', )
    parser.add_argument('-mt', '--model_type', type=str, default='effnet', )
    parser.add_argument('-mo', '--model_output', type=str, default='regression', )
    parser.add_argument('-rs', '--re_size', type=int, default=300, )
    parser.add_argument('-bs', '--batch_size', type=int, default=8, )
    parser.add_argument('-eps', '--epochs', type=int, default=100, )
    parser.add_argument('-wts', '--weights', type=bool, default=True, )
    parser.add_argument('-tta_f', '--tta_flag', type=bool, default=True, )
    args = parser.parse_args()
    main(args)
