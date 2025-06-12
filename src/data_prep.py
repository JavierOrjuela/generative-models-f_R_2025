"""
Code written by H.J.H for the paper:
Conditional Diffusion-Flow models for generating 3D cosmic
density fields: applications to 𝑓(𝑅) cosmologies
https://arxiv.org/abs/2502.17087

"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from pathlib import Path
import numpy as np
import pandas as pd
from volumentations import *
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
RANDOM=123


os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
gpus
print(tf.config.list_physical_devices('GPU'))

dir_root = ''
name_pandas= dir_root+'latin_hypercube_params.txt'


removes=[]
def min_values(df,image_File=''):
     try:
         image=np.load(image_File+df,allow_pickle=True)
         min_val=image.min()
         return min_val
     except:
         print("Error! Could not load encoder for feature ", Path(df).parts[0])
         removes.append(int(Path(df).parts[0]))
         return None
def max_values(df,image_File=''):
     try:
         image=np.load(image_File+df,allow_pickle=True)
         max_val=image.max()
         return max_val
     except:
         print("Error! Could not load encoder for feature ", Path(df).parts[0])
         return None

Train_LSS=pd.read_csv(dir_root+'Train_LSS.csv')
Validation_LSS=pd.read_csv(dir_root+'Validation_LSS.csv')
Test_LSS=pd.read_csv(dir_root+'Test_LSS.csv')


print("Length: Test_LSS", len(Test_LSS))
print("Length: Train_LSS", len(Train_LSS))
print("Length: Validation_LSS", len(Validation_LSS))

params=['Om', 'h', 'sigma8', 'fR0_scaled']
features= Train_LSS[params].values

num_features=len(params)
print('num_features', num_features)

scaler = MinMaxScaler()
scaler.fit(features)
print(scaler.data_max_)
print(scaler.data_min_)

removes=[]
min_val_64 = Train_LSS.filename_path.map(lambda x: min_values(x)).min()

max_val_64 = Train_LSS.filename_path.map(lambda x: max_values(x)).max()

print('min_val_64', min_val_64)
print('max_val_64', max_val_64)
outlier_clip=251.2
min_val_64_log=np.log(min_val_64+2.)
max_val_64_log=np.log(outlier_clip+2.)

print('min_val_64_log', min_val_64_log)
print('max_val_64_log', max_val_64_log)

def normalization_data_64(array):
    array= np.clip(array,-2., outlier_clip)
    array_log=np.log(array+2.)
    return (array_log-min_val_64_log)/(max_val_64_log-min_val_64_log)

def normalization_features(feat):
    return (feat-scaler.data_min_)/(scaler.data_max_-scaler.data_min_)

def normalization_features_mean(feat):
  return feat*(scaler.data_max_-scaler.data_min_)+scaler.data_min_
def normalization_features_var(feat):
  return feat*(scaler.data_max_-scaler.data_min_)**2

def get_augmentation():
    return Compose([
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        Flip(2, p=0.5),
        RandomRotate90((1, 2), p=0.5),
        GaussianNoise(var_limit=(0.001, 0.01), p=0.2),
    ], p=1.0)
aug = get_augmentation()

def load_arrays(path,dim='32'):
    if dim=='32':
        return None
    else:
        image_ = np.load(path.numpy().decode())
        image =normalization_data_64(image_)
        data = {'image': image}
        aug_data= aug(**data)
        image= np.clip(aug_data['image'],0,1)
    return image.reshape(image_.shape+(1,))


def tf_data_array(filenames1,features):
  [feat,]= tf.py_function(normalization_features, [features], [tf.float64])
  [image_64,] = tf.py_function(func = load_arrays,  inp = [filenames1,'64'], Tout = [tf.float64])
  image_64.set_shape((64,64,64,1,))
  feat.set_shape(features.shape)
  return image_64, feat



BATCH_SIZE = 16  # Adjusted batch size
total_iterations=Train_LSS.shape[0]//BATCH_SIZE

print(f'total_iterations: {total_iterations}')

params=['Om', 'h', 'sigma8', 'fR0_scaled']

def datasets_iteration(dataset_row, training=True):
    filename_64 = dataset_row['filename_path']
    features = dataset_row[params].values
    dataset = tf.data.Dataset.from_tensor_slices((filename_64, features))
    if training:
        dataset = dataset.shuffle(len(dataset_row), seed=RANDOM)
    dataset = dataset.map(tf_data_array).batch(BATCH_SIZE,drop_remainder=True)
    return dataset

Train_dataset = datasets_iteration(Train_LSS)
Validation_dataset = datasets_iteration(Validation_LSS, False)
Test_dataset = datasets_iteration(Test_LSS, False)

for i,j in Train_dataset.take(1):
    print(i.shape,j.shape)

for i,j in Validation_dataset.take(1):
    print(i.shape)

for i,j in Test_dataset.take(1):
    print(i.shape)

