import shutil
import pickle
import time
import datetime
import json
import os
import glob
import numpy as np
import keras
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers import Input
from keras.preprocessing import image
from keras.applications.densenet import DenseNet121

import hashlib


import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# import seaborn as sns
# import matplotlib.pyplot as plt


# config variables
if len(sys.argv) != 2:
    print('Usage: python train.py <path-to-config-file>')
    exit()

config = json.load(open(sys.argv[1]))

BASE_MODEL = config['model']
DATASET_PATH = config['dataset_path']
OUTPUT_PATH = os.path.join(config['output_path'], BASE_MODEL)

seed = 123456

if not os.path.exists('cache'):
    os.makedirs('cache')

cache_file = os.path.join('cache', '{}-{}.pickle'.format(BASE_MODEL,os.path.split(DATASET_PATH)[1]))

feature_cache = {}
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        feature_cache = pickle.load(f)
    print("[INFO] using cache...")


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.
    This function applies the "Inception" preprocessing which converts
    the RGB values from [0, 255] to [-1, 1].
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Preprocessed array.
    """
    x /= 128.
    x -= 1.
    return x.astype(np.float32)


# start time
print("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
start = time.time()

if BASE_MODEL == 'densenet121':
    model = DenseNet121(include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)), input_shape=(224, 224, 3), pooling='avg')
    image_size = (224, 224)
else:
    raise Exception('no such model')

print("[INFO] successfully loaded base model...")

all_images_hash = set()
# loop over all the labels in the folder
for label in os.listdir(DATASET_PATH):
    cur_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(cur_path):
        continue
    print("[INFO] processing label - " + label)
    for i, image_path in enumerate(glob.glob(os.path.join(cur_path, "*.jpg"))):
        hasher = hashlib.md5()
        with open(image_path, 'rb') as f:
            hasher.update(f.read())
        image_hash = hasher.hexdigest()
        all_images_hash.add(image_hash)
        # print(image_hash)

        if image_hash in feature_cache:
            print(".", end=' ', flush=True)
            continue

        img = image.load_img(image_path, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        flat = feature.flatten()
        # features.append(flat)
        # labels.append(label)

        feature_cache[image_hash] = {
            "feature": flat,
            "label": label
        }

        print(i, end=' ', flush=True)

    with open(cache_file, 'wb') as f:
        pickle.dump(feature_cache, f)
    print("\n[INFO] saving cache...")


# variables to hold features and labels
features = []
labels = []

for image_hash, value in feature_cache.items():
    if image_hash not in all_images_hash:
        continue
    features.append(value['feature'])
    labels.append(value['label'])

# encode the labels using LabelEncoder
le = LabelEncoder()
le_labels = le.fit_transform(labels)

# get the shape of training labels
print("[STATUS] training labels: {}".format(le_labels))
print("[STATUS] training labels shape: {}".format(le_labels.shape))


print("[INFO] training started...")
# split the training and testing data
(x_train, x_test, y_train, y_test) = train_test_split(np.array(features),
                                                      np.array(le_labels),
                                                      test_size=0.1,
                                                      random_state=seed)

print("[INFO] splitted train and test data...")
print("[INFO] train data  : {}".format(x_train.shape))
print("[INFO] test data   : {}".format(x_test.shape))
print("[INFO] train labels: {}".format(y_train.shape))
print("[INFO] test labels : {}".format(y_test.shape))

# use logistic regression as the model
print("[INFO] creating LR classifier...")
classifier = LogisticRegression(random_state=seed, multi_class='ovr')
classifier.fit(x_train, y_train)

# use rank-1 and rank-5 predictions
print("[INFO] evaluating classifier...")
rank_1 = 0
rank_5 = 0

# loop over test data
for (features, label) in zip(x_test, y_test):
    # predict the probability of each class label and
    # take the top-5 class labels
    predictions = classifier.predict_proba(np.atleast_2d(features))[0]
    predictions = np.argsort(predictions)[::-1][:5]

    # rank-1 prediction increment
    if label == predictions[0]:
        rank_1 += 1

    # rank-5 prediction increment
    if label in predictions:
        rank_5 += 1

# convert accuracies to percentages
rank_1 = (rank_1 / float(len(y_test))) * 100
rank_5 = (rank_5 / float(len(y_test))) * 100


def log(message, append=False):
    print(message)
    flag = 'w'
    if append:
        flag = 'a'
    with open(os.path.join(OUTPUT_PATH, 'result.txt'), flag) as f:
        f.write(message + '\n')


if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)

os.makedirs(OUTPUT_PATH)

# write the accuracies to file
log("Rank-1: {:.2f}%".format(rank_1))
log("Rank-5: {:.2f}%\n".format(rank_5), append=True)

# evaluate the model of test data
preds = classifier.predict(x_test)

# write the classification report to file
log("{}".format(classification_report(y_true=y_test, y_pred=preds, target_names=le.classes_)), append=True)

# display the confusion matrix
# print("[INFO] confusion matrix")

# plot the confusion matrix
# cm = confusion_matrix(y_test, preds)
# sns.heatmap(cm,
#             annot=True,
#             cmap="Set2")
# plt.show()

# dump model to file
print("[INFO] saving model...")

model.save(os.path.join(OUTPUT_PATH, 'model.h5'))
pickle.dump(classifier, open(os.path.join(
    OUTPUT_PATH, 'classifier.pickle'), 'wb'))

# save label mappings
with open(os.path.join(OUTPUT_PATH, 'label.txt'), 'w') as f:
    for i, label in enumerate(le.classes_):
        f.write('{}\t{}\n'.format(i, label))

# end time
end = time.time()
print("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
