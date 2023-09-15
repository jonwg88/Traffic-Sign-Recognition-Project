#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Import required libraries
import cv2
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join


# In[21]:


#Read the Train_data_label Excel file
train_data = pd.read_excel("Dropbox/Coding/AI and ML Bootcamp/Course 4 Deep Learning with Keras and TensorFlow/data/Train_data_label.xlsx")


# In[32]:


#Read train folder images and resize them to 30x30
#Create a NumPy array from resized images
base_path = "Dropbox/Coding/AI and ML Bootcamp/Course 4 Deep Learning with Keras and TensorFlow/data"
def read_and_resize(base_path, image_paths):
    train_resized = []
    for path in image_paths:
        full_path = base_path + "/" + path
        image = cv2.imread(full_path)
        image = cv2.resize(image, (30, 30))
        train_resized.append(image)
    return np.array(train_resized)
train_resized = read_and_resize(base_path, train_data['Path'].values)


# In[33]:


#Check number and dimensions of images
print(train_resized.shape)


# In[34]:


#Convert RGB images to grayscale
grayscale_img = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in train_resized]
#Check length of arrays to make sure they match
print(len(train_resized))
print(len(grayscale_img))


# In[35]:


#Save color image and grayscale image arrays in new columns in train_data
train_data['color_img_arr'] = [img for img in train_resized]
train_data['grayscale_img_arr'] = [img for img in grayscale_img]


# In[36]:


#Import 'test_data' for bar chart
test_data = pd.read_excel('Dropbox/Coding/AI and ML Bootcamp/Course 4 Deep Learning with Keras and TensorFlow/data/Test_data_label.xlsx')


# In[38]:


#Plot bar chart of classes from train and test data with actual class names
#Get value counts of classes in datasets and organize them by index
train_class_count = train_data['ClassId'].value_counts().sort_index()
test_class_count = test_data['ClassId'].value_counts().sort_index()
#Define class names in index order
class_names = ["Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)", 
               "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)", "No passing", "No passing vehicle over 3.5 tons", 
               "Right-of-way at the intersection", "Priority road", "Yield", "Stop", "No vehicles", "Vehicle > 3.5 tons prohibited", "No entry", "General caution", 
               "Dangerous curve left", "Dangerous curve right", "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", "Road work", "Traffic signals", 
               "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing", "End speed + passing limits", "Turn right ahead", 
               "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory", "End of no passing", 
               "End no passing vehicle > 3.5 tons"]


# In[43]:


#Plot classes from train data
plt.figure(figsize=(14,7))
train_class_count.plot(kind='bar')
plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=90)
plt.title('Train Data Class Distribution')
plt.xlabel('Traffic Signs')
plt.ylabel('Number of Images')
plt.show()


# In[44]:


#Plot classes from test data
plt.figure(figsize=(14,7))
test_class_count.plot(kind='bar')
plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=90)
plt.title('Test Data Class Distribution')
plt.xlabel('Traffic Signs')
plt.ylabel('Number of Images')
plt.show()


# In[47]:


#Prepare the model's training and testing data and convert to appropriate format and shape
#Normalize training images
train_color = train_resized.astype('float32') / 255
grayscale_img_array = np.array(grayscale_img)
train_grayscale = grayscale_img_array.astype('float32') / 255

#Reshape training grayscale images to have a depth dimension (make grayscale images 3-D to match color images)
train_grayscale = train_grayscale.reshape(-1, 30, 30, 1)


# In[48]:


#One-hot encode the training dataset labels
from tensorflow.keras.utils import to_categorical
train_labels = train_data['ClassId'].values
train_labels_encoded = to_categorical(train_labels, num_classes=43)


# In[49]:


#Split color and grayscale images into training and validation sets
from sklearn.model_selection import train_test_split
X_train_color, X_val_color, y_train_color, y_val_color = train_test_split(train_color, train_labels_encoded, test_size=0.2, random_state=42)
X_train_gray, X_val_gray, y_train_gray, y_val_gray = train_test_split(train_grayscale, train_labels_encoded, test_size=0.2, random_state=42)


# In[50]:


#Read test folder images and resize them to 30x30
#Create a NumPy array from resized images
base_path = "Dropbox/Coding/AI and ML Bootcamp/Course 4 Deep Learning with Keras and TensorFlow/data"
test_resized = read_and_resize(base_path, test_data['Path'].values)


# In[54]:


#Normalize test dataset, convert to grayscale, and one-hot encode labels
test_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in test_resized]
test_gray = np.array(test_gray)
test_color = test_resized.astype('float32') / 255
test_gray = test_gray.astype('float32') / 255
test_gray = test_gray.reshape(-1, 30, 30, 1)

test_labels = test_data['ClassId'].values
test_labels_encoded = to_categorical(test_labels, num_classes=43)


# In[55]:


#Create CNN model for color images
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

color_model = Sequential()

#Add convolutional, MaxPool, and dropout layers with relu activation function
color_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 3)))
color_model.add(MaxPooling2D((2, 2)))
color_model.add(Dropout(0.25))

color_model.add(Conv2D(64, (3, 3), activation='relu'))
color_model.add(MaxPooling2D((2, 2)))
color_model.add(Dropout(0.25))

color_model.add(Flatten())
color_model.add(Dense(256, activation='relu'))
color_model.add(Dropout(0.5))
color_model.add(Dense(43, activation='softmax'))


# In[56]:


#Create CNN model for grayscale images
grayscale_model = Sequential()

grayscale_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 1)))
grayscale_model.add(MaxPooling2D((2, 2)))
grayscale_model.add(Dropout(0.25))

grayscale_model.add(Conv2D(64, (3, 3), activation='relu'))
grayscale_model.add(MaxPooling2D((2, 2)))
grayscale_model.add(Dropout(0.25))

grayscale_model.add(Flatten())
grayscale_model.add(Dense(256, activation='relu'))
grayscale_model.add(Dropout(0.5))
grayscale_model.add(Dense(43, activation='softmax'))


# In[57]:


#Compile models with loss function as categorical cross-entropy and Adam optimizer
color_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
grayscale_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[58]:


#Prepare early stopping with two patience and validation loss monitoring
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=2)


# In[59]:


#Experiment with both models for five epochs with early stopping
color_history = color_model.fit(X_train_color, y_train_color, validation_data=(X_val_color, y_val_color), epochs=5, batch_size=64, callbacks=[early_stop])
grayscale_history = grayscale_model.fit(X_train_gray, y_train_gray, validation_data=(X_val_gray, y_val_gray), epochs=5, batch_size=64, callbacks=[early_stop])


# In[62]:


#Plot training and validation accuracy for both models
plt.figure(figsize=(12, 6))

#Plot color model accuracy
plt.subplot(1, 2, 1)
plt.plot(color_history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(color_history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Color Model Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

#Plot grayscale model accuracy
plt.subplot(1, 2, 2)
plt.plot(grayscale_history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(grayscale_history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Grayscale Model Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# In[65]:


#Observe precision, recall, and F1-score for all classes for both models
#Get predicted class labels
color_pred = color_model.predict(X_val_color)
color_pred_classes = np.argmax(color_pred, axis=1)
gray_pred = grayscale_model.predict(X_val_gray)
gray_pred_classes = np.argmax(gray_pred, axis=1)
true_classes = np.argmax(y_val_color, axis=1)

#Compute precision, recall, and F1-score
from sklearn.metrics import classification_report
color_report = classification_report(true_classes, color_pred_classes, target_names=class_names)
print("Classification report for color model:")
print(color_report)

grayscale_report = classification_report(true_classes, gray_pred_classes, target_names=class_names)
print("Classification report for grayscale model:")
print(grayscale_report)


# Both models had great classification across all classes. Nearly every precision value, recall score, and F1-score was 1.00 or just below 1.00. The lowest score was a 0.89 (a recall score for the 'Beware of ice/snow' class), but the majority of classes had scores exceeding 0.95 in all metrics. A significant amount of classes had perfect 1.00 scores in all metrics. The classes are excellent, and considering the purpose of the program, no improvement is needed.

# In[67]:


#Evaluate model on test set
color_test_loss, color_test_accuracy = color_model.evaluate(test_color, test_labels_encoded)
gray_test_loss, gray_test_accuracy = grayscale_model.evaluate(test_gray, test_labels_encoded)


# In[69]:


#Test set classification report
color_predictions = color_model.predict(test_color)
color_predicted_classes = np.argmax(color_predictions, axis=1)
grayscale_predictions = grayscale_model.predict(test_gray)
grayscale_predicted_classes = np.argmax(grayscale_predictions, axis=1)
test_classes = np.argmax(test_labels_encoded, axis=1)

print("Color model classification report:")
print(classification_report(test_classes, color_predicted_classes, target_names=class_names))

print("Grayscale model classification report:")
print(classification_report(test_classes, grayscale_predicted_classes, target_names=class_names))


# Both models performed well on the test set. The performance was noticeably worse on the test set than the validation set, and some classes had metrics significantly below 0.80. The classes were good overall but some performed relatively poorly. Each model had a few classes with disappointing F1-scores. The color model struggled with 'Double Curve' and the grayscale model struggled with 'Speed limit (20km/h)'. Both models showed poor performance when classifying 'Pedestrians' and 'Beware of ice/snow'. That said, both models had a fairly low loss value (0.2541 for the color model and 0.2813 for the grayscale model) and both models had high accuracy scores (0.9282 for the color model and 0.9317 for the grayscale model).

# **Compare color and grayscale models**
# The color model showed better performance on the validation set, and had lower training and validation loss and higher training and validation accuracy than the grayscale model. Both models had stellar classification reports, but the color model showed slightly better scores in the three metrics. Neither model showed signs of overfitting. The models performed roughly equivalently on the test set. The grayscale model had higher accuracy, but the color model had lower loss. The performance reports were similar enough that neither model appeared to be blatantly superior, though the color model had higher metrics overall with the two classes that both models struggled with ('Pedestrians' and 'Beware of ice/snow'). Both models performed admirably, and each had weaknesses in their classifications. However, the color model seemed slighly better overall, which makes sense considering that many traffic signs are color-coded.
