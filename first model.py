import os
import pandas as pd
import pickle
import seaborn as sns
from skimage.io import imread
from skimage.transform import resize
import numpy as np

# First model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Second model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Working on the data and showing a little bit of plots
Dir = r'C:\Users\yoavs\Desktop\another one\clf-data'
categories = ['empty', 'not_empty']

dataa = []

for category in categories:
    category_dir = os.path.join(Dir, category)
    for file in os.listdir(category_dir):
        image_path = os.path.join(category_dir, file)
        image_data = {'Category': category, 'Image': image_path}
        dataa.append(image_data)

df = pd.DataFrame(dataa)
print(df)
print(df.describe())

sns.countplot(x='Category', data=df)
plt.show()

df['Image Size'] = df['Image'].apply(lambda x: os.stat(x).st_size)
plt.figure(figsize=(8, 6))
sns.boxplot(x='Category', y='Image Size', data=df)
plt.title('Distribution of Image Sizes by Category')
plt.xlabel('Category')
plt.ylabel('Image Size')
plt.show()


df_numeric = df.drop('Category', axis=1)
sns.pairplot(df_numeric)
plt.show()


sns.pairplot(df, hue='Category')
plt.show()


sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
ax = sns.histplot(data=df, x='Image Size', hue='Category', kde=True)

plt.title('Distribution of Image Sizes', fontsize=16)
plt.xlabel('Image Size', fontsize=12)
plt.ylabel('Count', fontsize=12)

sns.despine()
plt.grid(True, linestyle='--', alpha=0.5)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, title='Category', fontsize=10, title_fontsize=12)

plt.show()

feature_stats = df.drop('Category', axis=1).describe()

print(feature_stats)

sns.set(style="ticks")
sns.pairplot(df, hue="Category")
plt.show()

# starting the real work
data = [resize(imread(os.path.join(Dir, category, file)), (15, 15)).flatten()
        for category_idx, category in enumerate(categories)
        for file in os.listdir(os.path.join(Dir, category))]

labels = [category_idx
          for category_idx, category in enumerate(categories)
          for file in os.listdir(os.path.join(Dir, category))]

data = np.asarray(data)
labels = np.asarray(labels)

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=0.5, shuffle=True, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, shuffle=True, stratify=y_train)

# define the model
Classifier = SVC()

HYPER_PARAMETERS = [{'gamma': [0.01, 0.001, 0.0001] , 'C' : [1, 10, 100, 1000]}]
Search = GridSearchCV(Classifier, HYPER_PARAMETERS)
#searching for the best parameters
Search.fit(X_train, y_train)

Best_Model = Search.best_estimator_
print(Best_Model)

y_valid_pred = Best_Model.predict(X_val)
valid_accuracy = accuracy_score(y_val, y_valid_pred)
print("Validation Accuracy: {:.2f}%".format(valid_accuracy * 100))

y_test_pred = Best_Model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

import seaborn as sns
from sklearn.metrics import confusion_matrix

accuracy = accuracy_score(y_test, y_test_pred)
report = classification_report(y_test, y_test_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(report)
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6,4))
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

pickle.dump(Best_Model, open('./model.h5', 'wb'))