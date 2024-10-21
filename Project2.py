# Importing the libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import cv2
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import warnings
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
warnings.filterwarnings('ignore')

tf.config.run_functions_eagerly(True)

kaggle_prefix = "/kaggle/input/" # Note: If we simply put empty string, then the entire code will run in google colab without any issues.

#  read in the annotations

data = pd.read_csv('/kaggle/input/celeba-dataset/list_attr_celeba.csv')

#  read in the landmarks
landmarks = pd.read_csv('/kaggle/input/celeba-dataset/list_landmarks_align_celeba.csv')

# read in the partitions to discover train, validation and test segments
partition = pd.read_csv('/kaggle/input/celeba-dataset/list_eval_partition.csv')

# selecting only male and bushy eyebrows features. We will also need image_id for unique identfication of these images
data = data[['image_id', 'Male', 'Bushy_Eyebrows']]

# transform targets to range (0,1)


data['Male'].replace({-1: 0}, inplace=True)
data['Bushy_Eyebrows'].replace({-1: 0}, inplace=True)

# perform an inner join of the result with the partition data frame on image_id to obtain integrated partitions
df = pd.merge(data, partition, on='image_id', how='inner')

train_df = df[df['partition'] == 0]
test_df = df[df['partition'] == 1]
val_df = df[df['partition'] == 2]

train_df = train_df.sample(n=10000, random_state=42)
train_df.head()
val_df = val_df.sample(n=1000, random_state=42)
val_df.head()
print (len(val_df))
test_df = test_df.sample(n=1000, random_state=42)
print(len(test_df))
print(test_df.head())

print(f"Attributes: \n{data.head(2)}\n\n")
#print(f"Landmarks: \n{landmarks.head(2)}\n\n")
print(f"partitions: \n{partition.head(2)}\n\n")

print(f"Training Data: \n{train_df.head(2)}\n\n")
print(f"Validation Data: \n{val_df.head(2)}\n\n")
print(f"Test Data: \n{test_df.head(2)}\n\n")

print(f"Lengths of train, validation and test partitions: {len(train_df), len(test_df), len(val_df)}")

# Data generator class - this is a key class that is used to batch the data so as to
# reduce compute time as well as to fit training segments into available memory
# Additionally it allows you to specify multiple targets for classification
# Also allows for image cropping
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.utils import np_utils

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, df, batch_size=32, dim=(218,178), n_channels=3, n_classes=2, shuffle=False):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size)) # Modified code here to include the last batch as well.

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # saves memory by batching
        df_temp = self.df.iloc[indexes].reset_index(drop=True)
        X, y = self.__data_generation(df_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, df_temp):
        # Adjust for the last batch which might be smaller
        current_batch_size = len(df_temp)
        X = np.empty((current_batch_size, *self.dim, self.n_channels))
        y = np.empty((current_batch_size, self.n_classes),dtype=int)

        for i, row in df_temp.iterrows():
                img_path ='/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/' + row[0]
                img = image.load_img(img_path, target_size=self.dim)
                img = image.img_to_array(img)
                img /= 255.0
                X[i,] = img
                y[i,] = row[1:self.n_classes+1]
        return X, y
    
# using vgg16 as feature extractor
vgg16 = tf.keras.applications.VGG16(input_shape=(218, 178, 3), include_top=False, weights='imagenet')
vgg16.trainable = False

# creating the model
model = tf.keras.Sequential([
    vgg16,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])
model.compile(optimizer=SGD(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
# creating the train, test and validation data generators
train_generator = DataGenerator(train_df, batch_size=32, dim=(218,178), n_channels=3, n_classes=2, shuffle=False)
test_generator = DataGenerator(test_df, batch_size=32, dim=(218,178), n_channels=3, n_classes=2, shuffle=False)
val_generator = DataGenerator(val_df, batch_size=32, dim=(218,178), n_channels=3, n_classes=2, shuffle=False)

# training the model
history = model.fit(train_generator, epochs=3, validation_data=val_generator)

print(model.evaluate(test_generator,batch_size=32))

prediction = model.predict(test_generator)
test_array=test_df.to_numpy()
for pred in prediction:
    print(pred)

test_size=10
correct=0
for i in range(test_size):
    pred_class1 = prediction[i,0]
    pred_class2= prediction[i,1]
    test_class1 = test_array[i,1]
    test_class2= test_array[i,2]
    print(f"{i}, {pred_class1:.2f}, {test_class1}, {pred_class2:.2f}, {test_class2}")
    if((np.where(pred_class1 > 0.5, 1,0))==test_class1
       and (np.where(pred_class2 > 0.5, 1,0))==test_class2): correct=correct+1
print(f"test accuracy is: {(correct/test_size):.1%}")   


# plotting the training and validation accuracy

# Note: Uncomment this code, to see the plot of accuracies of the above run.


# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='upper left')
# plt.show()

data = pd.read_csv(kaggle_prefix + 'celeba-dataset/list_attr_celeba.csv')

data = data[['image_id', 'Male', 'Arched_Eyebrows']]

# read in the partitions to discover train, validation and test segments
partition = pd.read_csv(kaggle_prefix + 'celeba-dataset/list_eval_partition.csv')

# perform an inner join of the result with the partition data frame on image_id to obtain integrated partitions
df = pd.merge(data, partition, on='image_id', how='inner')

train_df = df[df['partition'] == 0]
test_df = df[df['partition'] == 1]
val_df = df[df['partition'] == 2]

print(f"Attributes: \n{data.head(2)}\n\n")
print(f"partitions: \n{partition.head(2)}\n\n")
print(f"Merged: \n{df.head(2)}\n\n")

print(f"Training Data: \n{train_df.head(2)}\n\n")
print(f"Validation Data: \n{val_df.head(2)}\n\n")
print(f"Test Data: \n{test_df.head(2)}\n\n")

print(f"Lengths of train, validation and test partitions: {len(train_df), len(test_df), len(val_df)}")

# Data generator class - this is a key class that is used to batch the data so as to
# reduce compute time as well as to fit training segments into available memory
# Additionally it allows you to specify multiple targets for classification
# Also allows for image cropping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, df, batch_size=32, dim=(64,64), n_channels=3, n_classes=2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size)) # Modified code here to include the last batch as well.

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # saves memory by batching
        df_temp = self.df.iloc[indexes].reset_index(drop=True)
        X, y = self.__data_generation(df_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, df_temp):
        # Adjust for the last batch which might be smaller
        current_batch_size = len(df_temp)
        X = np.empty((current_batch_size, *self.dim, self.n_channels))
        y = np.empty((current_batch_size, self.n_classes), dtype=int)

        for i, row in df_temp.iterrows():
            try:
                img_path = kaggle_prefix + 'celeba-dataset/img_align_celeba/img_align_celeba/' + row[0]
                img = image.load_img(img_path, target_size=self.dim)
                img = image.img_to_array(img)
                img /= 255.0
                X[i,] = img
                y[i,] = row[1: self.n_classes + 1]
            except Exception as e:
                print(f"Error loading image: {e}")

        return X, y

#     def __data_generation(self, df_temp):
#         X = np.empty((self.batch_size, *self.dim, self.n_channels))
#         y = np.empty((self.batch_size, self.n_classes), dtype=int)
#         for i, row in enumerate(df_temp.values):
#             img = image.load_img(kaggle_prefix +  'celeba-dataset/img_align_celeba/img_align_celeba/' + row[0]) # extract image
#             # normally images would need to be cropped at this point, but in this case there is no need
#             # as we are using the aligned and cropped version.
#             img = img.resize(self.dim)
#             img = image.img_to_array(img) # this are the image pixels flattened into an array

#             #normalize the resized images
#             X[i,] = img / 255.0 # this is the set of normalized pixel values captured into a 2D array for sample in the batch

#             #specify the multiple targets now into your y vector
#             y[i,] = row[1: self.n_classes + 1] # there are two targets for R1

#         return X, y

# using vgg16 as feature extractor
vgg16 = tf.keras.applications.VGG16(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
vgg16.trainable = False

# creating the model
model = tf.keras.Sequential([
    vgg16,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# compiling the model
model.compile(optimizer=SGD(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# creating the train, test and validation data generators
train_generator = DataGenerator(train_df, batch_size=450, dim=(64,64), n_channels=3, n_classes=2, shuffle=True)
test_generator = DataGenerator(test_df, batch_size=450, dim=(64,64), n_channels=3, n_classes=2, shuffle=True)
val_generator = DataGenerator(val_df, batch_size=450, dim=(64,64), n_channels=3, n_classes=2, shuffle=True)

# training the model
history = model.fit(train_generator, epochs=1, validation_data=val_generator) # Note: Here, We are running for only 1 epoch. We can run for 10 epochs by changing the epochs value here.
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1) # Note: This will improve depedning on the number of epochs that you run. Here, we are running for only 1 epoch.

print(test_loss, test_accuracy)

data = pd.read_csv(kaggle_prefix + 'celeba-dataset/list_attr_celeba.csv')

landmarks = pd.read_csv(kaggle_prefix + 'celeba-dataset/list_landmarks_align_celeba.csv')
landmarks['eye_width'] = landmarks['righteye_x'] - landmarks['lefteye_x']
landmarks['eye_width_normalized'] = (landmarks['eye_width'] - landmarks['eye_width'].min()) / (landmarks['eye_width'].max() - landmarks['eye_width'].min()) # (x - xmin) / (xmax - xmin)
landmarks['eye_width_quartile'] = pd.qcut(landmarks['eye_width_normalized'], 4, labels=[1, 2, 3, 4])

df = pd.merge(data, landmarks, on='image_id', how='inner')


# Filtering only the required columns
df = df[['image_id', 'Male', 'eye_width_quartile']]

# read in the partitions to discover train, validation and test segments
partition = pd.read_csv(kaggle_prefix + 'celeba-dataset/list_eval_partition.csv')

# perform an inner join of the result with the partition data frame on image_id to obtain integrated partitions
df = pd.merge(df, partition, on='image_id', how='inner')

train_df = df[df['partition'] == 0]
test_df = df[df['partition'] == 1]
val_df = df[df['partition'] == 2]

print(f"Attributes: \n{data.head(2)}\n\n")
print(f"Landmarks: \n{landmarks.head(2)}\n\n")
print(f"partitions: \n{partition.head(2)}\n\n")

print(f"Training Data: \n{train_df.head(2)}\n\n")
print(f"Validation Data: \n{val_df.head(2)}\n\n")
print(f"Test Data: \n{test_df.head(2)}\n\n")

print(f"Lengths of train, validation and test partitions: {len(train_df), len(test_df), len(val_df)}")

# creating the train, test and validation data generators
train_generator = DataGenerator(train_df, batch_size=450, dim=(64,64), n_channels=3, n_classes=2, shuffle=True)
test_generator = DataGenerator(test_df, batch_size=450, dim=(64,64), n_channels=3, n_classes=2, shuffle=True)
val_generator = DataGenerator(val_df, batch_size=450, dim=(64,64), n_channels=3, n_classes=2, shuffle=True)

# using vgg16 as feature extractor
vgg16 = tf.keras.applications.VGG16(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
vgg16.trainable = False

# creating the model
model = tf.keras.Sequential([
    vgg16,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

# compiling the model
model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# training the model

history = model.fit(train_generator, epochs=3, validation_data=val_generator)

test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(test_loss, test_accuracy)


test_df[(test_df['Male'] == -1) & (test_df['eye_width_quartile'] == 1)].head(2)
test_generator_female_q1 = DataGenerator(test_df[(test_df['Male'] == -1) & (test_df['eye_width_quartile'] == 1)], batch_size=450, dim=(64,64), n_channels=3, n_classes=2, shuffle=True)

test_loss, test_accuracy = model.evaluate(test_generator_female_q1, verbose=1)
print(test_loss, test_accuracy)

test_generator_female_q4 = DataGenerator(test_df[(test_df['Male'] == -1) & (test_df['eye_width_quartile'] == 4)], batch_size=450, dim=(64,64), n_channels=3, n_classes=2, shuffle=True)

test_loss, test_accuracy = model.evaluate(test_generator_female_q4, verbose=1)
print(test_loss, test_accuracy)

test_generator_male_q1 = DataGenerator(test_df[(test_df['Male'] == 1) & (test_df['eye_width_quartile'] == 1)], batch_size=450, dim=(64,64), n_channels=3, n_classes=2, shuffle=True)

test_loss, test_accuracy = model.evaluate(test_generator_male_q1, verbose=1)
print(test_loss, test_accuracy)

test_generator_male_q4 = DataGenerator(test_df[(test_df['Male'] == 1) & (test_df['eye_width_quartile'] == 4)], batch_size=450, dim=(64,64), n_channels=3, n_classes=2, shuffle=True)

test_loss, test_accuracy = model.evaluate(test_generator_male_q4, verbose=1)
print(test_loss, test_accuracy)

data = pd.read_csv(kaggle_prefix + 'celeba-dataset/list_attr_celeba.csv')

landmarks = pd.read_csv(kaggle_prefix + 'celeba-dataset/list_landmarks_align_celeba.csv')
landmarks['mouth_width'] = landmarks['rightmouth_x'] - landmarks['leftmouth_x']
landmarks['mouth_width_normalized'] = (landmarks['mouth_width'] - landmarks['mouth_width'].min()) / (landmarks['mouth_width'].max() - landmarks['mouth_width'].min()) # (x - xmin) / (xmax - xmin)
landmarks['mouth_width_quartile'] = pd.qcut(landmarks['mouth_width_normalized'], 4, labels=[1, 2, 3, 4])

df = pd.merge(data, landmarks, on='image_id', how='inner')


# Filtering only the required columns
df = df[['image_id', 'Smiling', 'mouth_width_quartile']]

# read in the partitions to discover train, validation and test segments
partition = pd.read_csv(kaggle_prefix + 'celeba-dataset/list_eval_partition.csv')

# perform an inner join of the result with the partition data frame on image_id to obtain integrated partitions
df = pd.merge(df, partition, on='image_id', how='inner')

train_df = df[df['partition'] == 0]
test_df = df[df['partition'] == 1]
val_df = df[df['partition'] == 2]

# creating the train, test and validation data generators
train_generator = DataGenerator(train_df, batch_size=450, dim=(64,64), n_channels=3, n_classes=2, shuffle=True)
test_generator = DataGenerator(test_df, batch_size=450, dim=(64,64), n_channels=3, n_classes=2, shuffle=True)
val_generator = DataGenerator(val_df, batch_size=450, dim=(64,64), n_channels=3, n_classes=2, shuffle=True)

print(f"Attributes: \n{data.head(2)}\n\n")
print(f"Landmarks: \n{landmarks.head(2)}\n\n")
print(f"partitions: \n{partition.head(2)}\n\n")

print(f"Training Data: \n{train_df.head(2)}\n\n")
print(f"Validation Data: \n{val_df.head(2)}\n\n")
print(f"Test Data: \n{test_df.head(2)}\n\n")
print(f"Lengths of train, validation and test partitions: {len(train_df), len(test_df), len(val_df)}")


# using vgg16 as feature extractor
vgg16 = tf.keras.applications.VGG16(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
vgg16.trainable = False

# creating the model
model = tf.keras.Sequential([
    vgg16,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# compiling the model
model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# training the model
history = model.fit(train_generator, epochs=1, validation_data=val_generator)

test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(test_loss, test_accuracy)

test_df[(test_df['Smiling'] == 1) & (test_df['mouth_width_quartile'] == 4)].head(2)

test_generator_smiling_q4 = DataGenerator(test_df[(test_df['Smiling'] == 1) & (test_df['mouth_width_quartile'] == 4)], batch_size=450, dim=(64,64), n_channels=3, n_classes=2, shuffle=True)

test_loss, test_accuracy = model.evaluate(test_generator_smiling_q4, verbose=1)
print(test_loss, test_accuracy)

test_generator_smiling_q1 = DataGenerator(test_df[(test_df['Smiling'] == 1) & (test_df['mouth_width_quartile'] == 1)], batch_size=450, dim=(64,64), n_channels=3, n_classes=2, shuffle=True)

test_loss, test_accuracy = model.evaluate(test_generator_smiling_q1, verbose=1)
print(test_loss, test_accuracy)