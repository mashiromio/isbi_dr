# isbi_dr

## train.py
`--dataset_path` is dataset root path, default is `./data`  
`--model_path` is model's saving path, default is `./models`  
`--model_type` is traing model type. There are two model: EfficientNetB5 and InceptionResNetV2, default is `effnet`  
`--model_output` is modle's top layer type, default is `regression`  
`--re_size` is resizing image after Preprocessing, default is `300`  
`--batch_size`, default is `8`  
`--epochs`, default is `100`  
`--weights` is whether using pre-training weights or not, default is `True`  
`--tta_flag` is whether using TTA or not, default is `True`  

Dataset file path structure is like following :  
```
data/
    aptos/
        train_images/
            0a4e1a29ffff.png
            ...
        train.csv
    eyepacs/
        train/
            10_left.jpeg
            ...
        trainLabels.csv
    isbi/
        regular-fundus-training/
            1/
                1_l1.jpg
                ...
            ...
            regular-fundus-training.csv
        
        regular-fundus-validation/
            265/
                265_l1.jpg
                ...
            ...
            regular-fundus-validation.csv
```
ISBI dataset download from  
https://isbi.deepdr.org/download.html  
  
APTOS dataset download from  
https://www.kaggle.com/c/aptos2019-blindness-detection/data  
  
EyePACS dataset download from  
https://www.kaggle.com/c/diabetic-retinopathy-detection/data
  
EfficientNetB5 pre-training weights download from  
https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b5_imagenet_1000_notop.h5  
  
InceptionResNetV2 pre-training weights download from  
https://www.kaggle.com/keras/inceptionresnetv2  
  
pre-training weights file path structure is like following :  
```
download_models/
    efficientnet-b5_imagenet_1000_notop.h5
    inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5
```
## valid.py
No command-line arguments.  
Should put trained models in `./all_models` folder. Like following :  
```
all_models/
    EfficientNetB5-1.h5
    EfficientNetB5-2.h5
    all_models/EfficientNetB5-3.h5
    all_models/EfficientNetB5-4.h5
    InceptionResNetV2-1.h5'
    InceptionResNetV2-2.h5
```
When run `valid.py`, default validation dataset is `./data/isbi`, file path structure is like following :  
```
data/
    isbi/
        regular-fundus-validation/
            265/
                265_l1.jpg
                ...
            ...
            regular-fundus-validation.csv
```
## test.py
No command-line arguments.  
Should put trained models in `./all_models` folder. Like following :  
```
all_models/
    EfficientNetB5-1.h5
    EfficientNetB5-2.h5
    all_models/EfficientNetB5-3.h5
    all_models/EfficientNetB5-4.h5
    InceptionResNetV2-1.h5'
    InceptionResNetV2-2.h5
```
When run `test.py`, default test dataset is `./data/test`, file path structure is like following :  
```
data/
    test/
        347/
            347_l1.jpg
            ...
        ...
```
And generating `Challenge1_upload.csv` file in `./upload`. `./upload` floder will be maked if it is not existing. Like following :  
```
upload/
    Challenge1_upload.csv
```
