import pickle
import os
import numpy as np
import sys
import keras
from keras.preprocessing import image


if len(sys.argv) != 3:
    print('Usage:')
    print('    python predict.py <path-to-model> <path-to-image>')
    exit()

MODEL_PATH = sys.argv[1]
predict_image_path = sys.argv[2]

print("[INFO] predicting {}...".format(predict_image_path))


print("[INFO] loading model...")

model = keras.models.load_model(os.path.join(MODEL_PATH, 'model.h5'))
image_size = (224, 224)

with open(os.path.join(MODEL_PATH, 'classifier.pickle'), 'rb') as f:
    classifier = pickle.load(f)

print("[INFO] successfully loaded model...")

index2label = {}
for line in open(os.path.join(MODEL_PATH, 'label.txt')):
    if line.strip() == '':
        continue
    index, label = line.strip().split('\t')
    index2label[int(index)] = label

img = image.load_img(predict_image_path, target_size=image_size)
img = image.img_to_array(img)
x = (img / 128) - 1
feature = model.predict(np.expand_dims(x, axis=0))
predictions = classifier.predict_proba(feature.reshape(1, -1))[0]

label_pred = [index2label[x] for x in np.argsort(predictions)[::-1]]
prob_pred = np.sort(predictions)[::-1]

for label, prob in zip(label_pred[:5], prob_pred[:5]):
    print('{:16}{}'.format(label, prob))
