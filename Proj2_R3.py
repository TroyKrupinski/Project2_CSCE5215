import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to dataset (adjusted for your local environment)
base_path = "C:/Users/dunke/Desktop/New folder/CelebA/"
annotations_path = os.path.join(base_path, 'Anno')
eval_path = os.path.join(base_path, 'Eval')
images_path = os.path.join(base_path, 'Img/img_align_celeba')

# Load attributes (Male, Young, and Smiling)
def load_attributes():
    data_path = os.path.join(annotations_path, 'list_attr_celeba.txt')
    data = pd.read_csv(data_path, sep=r'\s+', skiprows=1)
    data = data.reset_index()
    data = data.rename(columns={'index': 'image_id'})
    return data[['image_id', 'Male', 'Young', 'Smiling']]

# Load the partition data
def load_partitions():
    partition_path = os.path.join(eval_path, 'list_eval_partition.txt')
    partition = pd.read_csv(partition_path, sep=r'\s+', header=None, names=['image_id', 'partition'])
    return partition

# Load landmarks (for mouth and eye width calculation)
def load_landmarks():
    landmarks_path = os.path.join(annotations_path, 'list_landmarks_align_celeba.txt')
    landmarks = pd.read_csv(landmarks_path, sep=r'\s+', skiprows=1)
    landmarks = landmarks.reset_index()
    landmarks = landmarks.rename(columns={'index': 'image_id'})
    landmarks['mouth_width'] = landmarks['rightmouth_x'] - landmarks['leftmouth_x']
    landmarks['eye_width'] = landmarks['righteye_x'] - landmarks['lefteye_x']
    return landmarks[['image_id', 'mouth_width', 'eye_width']]

# Merge data
def merge_data(attributes, partition, landmarks):
    attributes['Male'] = attributes['Male'].replace({-1: 0})
    attributes['Young'] = attributes['Young'].replace({-1: 0})
    attributes['Smiling'] = attributes['Smiling'].replace({-1: 0})
    df = pd.merge(attributes, partition, on='image_id', how='inner')
    df = pd.merge(df, landmarks, on='image_id', how='inner')
    return df

# Split data into training, validation, and test sets
def split_data(df):
    train_df = df[df['partition'] == 0]
    test_df = df[df['partition'] == 1]
    val_df = df[df['partition'] == 2]
    return train_df, test_df, val_df

# Sample data to reduce training time
def sample_data(train_df, val_df, test_df, train_size=15000, val_size=1500, test_size=1500):
    train_df = train_df.sample(n=train_size, random_state=42)
    val_df = val_df.sample(n=val_size, random_state=42)
    test_df = test_df.sample(n=test_size, random_state=42)
    return train_df, val_df, test_df

# Data generator for image batches
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=64, dim=(128, 128), n_channels=3, targets=['Smiling'], shuffle=False):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.n_channels = n_channels
        self.targets = targets
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        df_temp = self.df.iloc[indexes].reset_index(drop=True)
        X, y = self.__data_generation(df_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, df_temp):
        X = np.empty((len(df_temp), *self.dim, self.n_channels))
        y = {target: np.empty((len(df_temp), 1), dtype=int) for target in self.targets}

        for i, row in df_temp.iterrows():
            img_path = os.path.join(images_path, row['image_id'])
            img = image.load_img(img_path, target_size=self.dim)
            img = image.img_to_array(img)
            img /= 255.0
            X[i,] = img
            for target in self.targets:
                y[target][i,] = row[target]
        
        return X, y

# Create a simple gender classification model for R1
def build_model_gender(input_shape):
    vgg16 = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    vgg16.trainable = False  # Freeze the VGG16 layers

    inputs = Input(shape=input_shape)
    x = vgg16(inputs, training=False)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    gender_output = Dense(1, activation='sigmoid', name='gender')(x)

    model = Model(inputs=inputs, outputs=[gender_output])
    model.compile(optimizer=SGD(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision')])
    
    return model

# Create multi-target model for gender and age (R2)
def build_model_multitarget(input_shape):
    vgg16 = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    for layer in vgg16.layers[:15]:
        layer.trainable = False  # Keep lower layers frozen

    inputs = Input(shape=input_shape)
    x = vgg16(inputs, training=True)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    
    # Output for gender (Male/Female)
    gender_output = Dense(1, activation='sigmoid', name='gender')(x)

    # Output for age (Young/Old)
    age_output = Dense(1, activation='sigmoid', name='age')(x)

    model = Model(inputs=inputs, outputs=[gender_output, age_output])
    model.compile(optimizer=SGD(learning_rate=0.001), 
                  loss={'gender': 'binary_crossentropy', 'age': 'binary_crossentropy'}, 
                  metrics={'gender': 'accuracy', 'age': 'accuracy'})
    
    return model

# Preprocess the CelebA dataset with landmarks
def preprocess_celeba_dataset():
    attributes = load_attributes()
    partition = load_partitions()
    landmarks = load_landmarks()
    df = merge_data(attributes, partition, landmarks)
    train_df, val_df, test_df = split_data(df)
    train_df, val_df, test_df = sample_data(train_df, val_df, test_df, train_size=15000, val_size=1500, test_size=1500)
    return train_df, val_df, test_df

# R1: Train and evaluate gender classification model
def r1_gender_classification():
    train_df, val_df, test_df = preprocess_celeba_dataset()

    # Initialize data generators
    train_generator = DataGenerator(train_df, batch_size=64, dim=(128, 128), n_channels=3, targets=['Male'], shuffle=True)
    val_generator = DataGenerator(val_df, batch_size=64, dim=(128, 128), n_channels=3, targets=['Male'], shuffle=True)
    test_generator = DataGenerator(test_df, batch_size=64, dim=(128, 128), n_channels=3, targets=['Male'], shuffle=False)

    # Build model
    model = build_model_gender(input_shape=(128, 128, 3))
    model.fit(train_generator, epochs=3, validation_data=val_generator)
    
    # Evaluate model on the test set
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"R1 (Gender Classification) Results - Loss: {test_loss}, Accuracy: {test_accuracy}")
    
    return test_loss, test_accuracy

# R2: Train and evaluate gender and age classification model
def r2_gender_age_classification():
    train_df, val_df, test_df = preprocess_celeba_dataset()

    # Initialize data generators
    train_generator = DataGenerator(train_df, batch_size=64, dim=(128, 128), n_channels=3, targets=['Male', 'Young'], shuffle=True)
    val_generator = DataGenerator(val_df, batch_size=64, dim=(128, 128), n_channels=3, targets=['Male', 'Young'], shuffle=True)
    test_generator = DataGenerator(test_df, batch_size=64, dim=(128, 128), n_channels=3, targets=['Male', 'Young'], shuffle=False)

    # Build multi-target model for gender and age
    model = build_model_multitarget(input_shape=(128, 128, 3))
    model.fit(train_generator, epochs=3, validation_data=val_generator)
    
    # Evaluate model on the test set
    test_results = model.evaluate(test_generator)
    combined_loss = test_results[0]  # Combined loss
    gender_accuracy = test_results[1]  # Gender accuracy
    age_accuracy = test_results[3]  # Age accuracy

    print(f"R2 (Gender and Age Classification) Results - Combined Loss: {combined_loss}, Gender Accuracy: {gender_accuracy}, Age Accuracy: {age_accuracy}")
    
    return combined_loss, gender_accuracy, age_accuracy

def build_model_smiling(input_shape):
    vgg16 = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    vgg16.trainable = False  # Freeze the VGG16 layers

    inputs = Input(shape=input_shape)
    x = vgg16(inputs, training=False)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    smiling_output = Dense(1, activation='sigmoid', name='smiling')(x)

    model = Model(inputs=inputs, outputs=[smiling_output])
    model.compile(optimizer=SGD(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision')])
    
    return model


# R3: Mouth Width and Eye Width Classification
def r3_mouth_and_eye_classification():
    # Preprocess the dataset
    train_df, val_df, test_df = preprocess_celeba_dataset()

    # Quartile Calculation for mouth width and eye width
    train_df['mouth_width_q'] = pd.qcut(train_df['mouth_width'], 4, labels=[1, 2, 3, 4])
    train_df['eye_width_q'] = pd.qcut(train_df['eye_width'], 4, labels=[1, 2, 3, 4])

    # (a) Train models for Q1 and Q4 for mouth width (Smiling)
    print("\n=== (a) Mouth Width Quartile 1 (Q1) vs Quartile 4 (Q4) ===")
    q1_train = train_df[train_df['mouth_width_q'] == 1]
    non_q1_train = train_df[train_df['mouth_width_q'] != 1]
    q4_train = train_df[train_df['mouth_width_q'] == 4]
    non_q4_train = train_df[train_df['mouth_width_q'] != 4]

    model_q1 = build_model_smiling(input_shape=(128, 128, 3))
    model_non_q1 = build_model_smiling(input_shape=(128, 128, 3))
    model_q4 = build_model_smiling(input_shape=(128, 128, 3))
    model_non_q4 = build_model_smiling(input_shape=(128, 128, 3))

    train_q1_generator = DataGenerator(q1_train, batch_size=64, dim=(128, 128), n_channels=3, targets=['Smiling'], shuffle=True)
    train_non_q1_generator = DataGenerator(non_q1_train, batch_size=64, dim=(128, 128), n_channels=3, targets=['Smiling'], shuffle=True)
    train_q4_generator = DataGenerator(q4_train, batch_size=64, dim=(128, 128), n_channels=3, targets=['Smiling'], shuffle=True)
    train_non_q4_generator = DataGenerator(non_q4_train, batch_size=64, dim=(128, 128), n_channels=3, targets=['Smiling'], shuffle=True)
    test_generator = DataGenerator(test_df, batch_size=64, dim=(128, 128), n_channels=3, targets=['Smiling'], shuffle=False)

    model_q1.fit(train_q1_generator, epochs=3)
    model_non_q1.fit(train_non_q1_generator, epochs=3)
    model_q4.fit(train_q4_generator, epochs=3)
    model_non_q4.fit(train_non_q4_generator, epochs=3)

    # Evaluate models on the test set
    q1_results = model_q1.evaluate(test_generator)
    non_q1_results = model_non_q1.evaluate(test_generator)
    q4_results = model_q4.evaluate(test_generator)
    non_q4_results = model_non_q4.evaluate(test_generator)

    # (b) Compute precision scores for "Smiling"
    print("\n=== (b) Mouth Width Sensitivity Analysis ===")
    q1_predictions = model_q1.predict(test_generator)
    non_q1_predictions = model_non_q1.predict(test_generator)
    q4_predictions = model_q4.predict(test_generator)
    non_q4_predictions = model_non_q4.predict(test_generator)

    test_smiling_labels = test_df['Smiling'].values
    P1 = precision_score(test_smiling_labels, np.where(q1_predictions > 0.5, 1, 0))
    P2 = precision_score(test_smiling_labels, np.where(non_q1_predictions > 0.5, 1, 0))
    P3 = precision_score(test_smiling_labels, np.where(q4_predictions > 0.5, 1, 0))
    P4 = precision_score(test_smiling_labels, np.where(non_q4_predictions > 0.5, 1, 0))

    M1 = abs(P1 - P2)
    M2 = abs(P3 - P4)

    print(f"Precision for Q1 model (Smiling): {P1}")
    print(f"Precision for non-Q1 model (Smiling): {P2}")
    print(f"M1 (Difference for Q1 vs non-Q1): {M1}")
    print(f"Precision for Q4 model (Smiling): {P3}")
    print(f"Precision for non-Q4 model (Smiling): {P4}")
    print(f"M2 (Difference for Q4 vs non-Q4): {M2}")

    # (c) Train models for Q1 and Q4 for eye width (Female classification)
    print("\n=== (c) Eye Width Quartile 1 (Q1) vs Quartile 4 (Q4) for Female Classification ===")
    q1_train_female = train_df[(train_df['eye_width_q'] == 1) & (train_df['Male'] == 0)]
    q4_train_female = train_df[(train_df['eye_width_q'] == 4) & (train_df['Male'] == 0)]
    non_q1_train_female = train_df[(train_df['eye_width_q'] != 1) & (train_df['Male'] == 0)]
    non_q4_train_female = train_df[(train_df['eye_width_q'] != 4) & (train_df['Male'] == 0)]

    model_q1_female = build_model_smiling(input_shape=(128, 128, 3))
    model_q4_female = build_model_smiling(input_shape=(128, 128, 3))
    model_non_q1_female = build_model_smiling(input_shape=(128, 128, 3))
    model_non_q4_female = build_model_smiling(input_shape=(128, 128, 3))

    train_q1_female_generator = DataGenerator(q1_train_female, batch_size=64, dim=(128, 128), n_channels=3, targets=['Smiling'], shuffle=True)
    train_q4_female_generator = DataGenerator(q4_train_female, batch_size=64, dim=(128, 128), n_channels=3, targets=['Smiling'], shuffle=True)
    train_non_q1_female_generator = DataGenerator(non_q1_train_female, batch_size=64, dim=(128, 128), n_channels=3, targets=['Smiling'], shuffle=True)
    train_non_q4_female_generator = DataGenerator(non_q4_train_female, batch_size=64, dim=(128, 128), n_channels=3, targets=['Smiling'], shuffle=True)

    model_q1_female.fit(train_q1_female_generator, epochs=3)
    model_q4_female.fit(train_q4_female_generator, epochs=3)
    model_non_q1_female.fit(train_non_q1_female_generator, epochs=3)
    model_non_q4_female.fit(train_non_q4_female_generator, epochs=3)

    # Evaluate female models on the test set
    q1_female_results = model_q1_female.evaluate(test_generator)
    non_q1_female_results = model_non_q1_female.evaluate(test_generator)
    q4_female_results = model_q4_female.evaluate(test_generator)
    non_q4_female_results = model_non_q4_female.evaluate(test_generator)

    # (d) Compute precision scores for "Smiling" in females
    print("\n=== (d) Eye Width Sensitivity Analysis for Female Classification ===")
    q1_female_predictions = model_q1_female.predict(test_generator)
    non_q1_female_predictions = model_non_q1_female.predict(test_generator)
    q4_female_predictions = model_q4_female.predict(test_generator)
    non_q4_female_predictions = model_non_q4_female.predict(test_generator)

    P1_female = precision_score(test_smiling_labels, np.where(q1_female_predictions > 0.5, 1, 0))
    P2_female = precision_score(test_smiling_labels, np.where(non_q1_female_predictions > 0.5, 1, 0))
    P3_female = precision_score(test_smiling_labels, np.where(q4_female_predictions > 0.5, 1, 0))
    P4_female = precision_score(test_smiling_labels, np.where(non_q4_female_predictions > 0.5, 1, 0))

    M1_female = abs(P1_female - P2_female)
    M2_female = abs(P3_female - P4_female)

    print(f"Precision for Q1 female model (Smiling): {P1_female}")
    print(f"Precision for non-Q1 female model (Smiling): {P2_female}")
    print(f"M1 (Difference for Q1 vs non-Q1 for Female): {M1_female}")
    print(f"Precision for Q4 female model (Smiling): {P3_female}")
    print(f"Precision for non-Q4 female model (Smiling): {P4_female}")
    print(f"M2 (Difference for Q4 vs non-Q4 for Female): {M2_female}")




# Run the full classification process and print results
if __name__ == "__main__":
    print("=== Running R1 (Gender Classification) ===")
    r1_loss, r1_accuracy = r1_gender_classification()

    print("\n=== Running R2 (Gender and Age Classification) ===")
    r2_combined_loss, r2_gender_accuracy, r2_age_accuracy = r2_gender_age_classification()

    print("\n=== Running R3 (Mouth Width and Eye Width Classification) ===")
    r3_mouth_and_eye_classification()

    # Final results summary
    print("\n=== Final Results Summary ===")
    print(f"Results for R1 (Gender Classification):")
    print(f"Gender Accuracy: {r1_accuracy}, Gender Loss: {r1_loss}")
    
    print(f"\nResults for R2 (Gender and Age Classification):")
    print(f"Gender Accuracy: {r2_gender_accuracy}, Age Accuracy: {r2_age_accuracy}, Combined Loss: {r2_combined_loss}")