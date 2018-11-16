import coremltools
import pickle
import os
import shutil

MODEL_PATH = 'output/flowers17/densenet'
OUTPUT_PATH = 'mlmodel/densenet'

if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)

os.makedirs(OUTPUT_PATH)

print('[INFO] loading classifier model...')
with open(os.path.join(MODEL_PATH, 'classifier.pickle'), 'rb') as f:
    classifier = pickle.load(f)
print('[INFO] loaded...')

print('[INFO] saving model...')
model = coremltools.converters.sklearn.convert(classifier)
model.author = 'Lu Yiming'
model.save(os.path.join(OUTPUT_PATH, 'LR.mlmodel'))
print('[INFO] saved...')

print('[INFO] loading cnn model...')
model = coremltools.converters.keras.convert(
    os.path.join(MODEL_PATH, 'model.h5'),
    image_input_names='input1',
    image_scale=1/128,
    red_bias=-1,
    green_bias=-1,
    blue_bias=-1)
model.author = 'Lu Yiming'
print('[INFO] loaded...')

print('[INFO] saving model...')
model.save(os.path.join(OUTPUT_PATH, 'DenseNet.mlmodel'))
print('[INFO] saved...')

