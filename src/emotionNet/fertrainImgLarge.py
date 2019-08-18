import sys, os,random,cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing.image import load_img,img_to_array

num_features = 64
num_labels = 8
batch_size = 20
epochs = 100
width, height = 64, 64
emotion = ['Happy','Sad','Surprise','Fear','Disgust','Anger','Contempt','None']

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return sorted(images)

def load_batch_image(img_path, target_size=(width, height)):

    im = cv2.resize(cv2.imread(os.path.join(imgRoot,img_path),0), target_size)

    return im[...,np.newaxis]/255.0*2-1

def trainDataset(imgList, batch_num, batch_size):
    while True:
        classes = len(emotion)
        max_len =  batch_size* batch_num

        X_samples = []
        for item in emotion:
            X_samples = np.append(X_samples,np.random.choice(imgList[item],batch_num * batch_size))

        max_len *= classes
        ind = np.arange(max_len);np.random.shuffle(ind)
        X_samples = X_samples[ind]
        y_samples = np.repeat(np.eye(classes),batch_num*batch_size,axis=0)[ind]
        X_batches = np.split(X_samples, batch_num)
        y_batches = np.split(y_samples, batch_num)

        for i in range(len(X_batches)):
            x = np.array(list(map(load_batch_image, X_batches[i])))
            y = np.array(y_batches[i])
            yield (x,y)

def valDataset(X, batch_size):
    while True:
        X_samples,y_samples = X['names'],X['label']
        batch_num = int(len(X_samples) / batch_size)
        max_len = batch_num * batch_size

        X_samples = np.array(X_samples[:max_len])
        y_samples = np.array(y_samples[:max_len])

        #print(max_len,batch_num)
        X_batches = np.split(X_samples, batch_num)
        y_batches = np.split(y_samples, batch_num)

        for i in range(len(X_batches)):
            x = np.array(list(map(load_batch_image, X_batches[i])))
            y = np.array(y_batches[i])
            yield (x,y)


#desinging the CNN
model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height,1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))

#model.summary()

#Compliling the model with adam optimixer and categorical crossentropy loss
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
imgRoot = '/media/anpei/win/Dataset/emotion/large/images'
root = '/home/anpei/Projects/face/fer2013/data/AffectNet/'
trainList = np.load(os.path.join(root,'training2.npy'),allow_pickle=True);trainList = trainList[()]
valList = np.load(os.path.join(root,'val2.npy'),allow_pickle=True);valList = valList[()]


train_steps = 1000
val_steps = int(len(valList['names']) / batch_size)

#training the model
model.fit_generator(
  trainDataset(trainList, train_steps, batch_size),
  epochs=epochs, 
  steps_per_epoch=train_steps,
  validation_data=valDataset(valList, batch_size),
  validation_steps=val_steps)

#saving the  model to be used later
fer_json = model.to_json()
with open("./checkpoints/large.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("./checkpoints/large.h5")
print("Saved model to disk")
