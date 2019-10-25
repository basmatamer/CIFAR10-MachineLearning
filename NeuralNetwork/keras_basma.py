
# coding: utf-8

# In[1]:


import numpy as np
import sys  
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.constraints import maxnorm
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import keras
import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint



# In[2]:


#adjust data
(X_train, y_train), (X_test, y_test) = cifar10.load_data() 

num_train, height, width, depth = X_train.shape #50,000
num_test = X_test.shape[0] # 10,000
num_classes = np.unique(y_train).shape[0] # 10

X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')
X_train /= np.max(X_train) #Normalize
X_test /= np.max(X_test)   #Normalize
Y_train = np_utils.to_categorical(y_train, num_classes) 
Y_test = np_utils.to_categorical(y_test, num_classes) 


# In[3]:


#set parameters
batch_size = 25
num_epochs = 400 
seed=7
kernel_size_1 = 3
kernel_size_2 = 1
pool_size = 2 
conv_depth_1 = 150 
conv_depth_2 = 100 
conv_depth_3 = 75 
drop_prob_1 = 0.25 
drop_prob_2 = 0.4 
hidden_size_1 = 750
hidden_size_2 = 500
hidden_size_3 = 250

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'



# In[4]:

model = Sequential()
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False) # randomly flip images
model.add(Conv2D(conv_depth_1, (kernel_size_1, kernel_size_1), input_shape=(32, 32, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(conv_depth_2, (kernel_size_1, kernel_size_1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(drop_prob_2))


model.add(Conv2D(conv_depth_3, (kernel_size_1, kernel_size_1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(conv_depth_3, (kernel_size_1, kernel_size_1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(drop_prob_2))

model.add(Conv2D(conv_depth_2 , (kernel_size_1, kernel_size_1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(conv_depth_2, (kernel_size_1, kernel_size_1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(drop_prob_1))

model.add(Flatten())

model.add(Dense(hidden_size_1, kernel_constraint=maxnorm(3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop_prob_2))

model.add(Dense(hidden_size_2, kernel_constraint=maxnorm(3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop_prob_2))

model.add(Dense(hidden_size_3, kernel_constraint=maxnorm(3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(drop_prob_2))


model.add(Dense(num_classes, activation='softmax'))


# In[ ]:

decay = 0.001/num_epochs
sgd = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
datagen.fit(X_train)
print(model.summary())
model_save = ModelCheckpoint(filepath='weights.hdf5' ,save_best_only=True ,monitor='val_loss', mode='min',verbose=1)


# In[ ]:

np.random.seed(seed)
X_train=X_train[:40000]
Y_train=Y_train[:40000]
X_val=X_train[40000:50000]
Y_val=Y_train[40000:50000]
ep_num = 200
with tf.device('/gpu:0'):
    model.fit_generator(datagen.flow(X_train, Y_train,batch_size=batch_size),
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            epochs=ep_num,
                            validation_data=(X_val,Y_val),workers=8 , verbose=1,callbacks=[model_save])


# In[ ]:

scores = model.evaluate(X_test, Y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:



