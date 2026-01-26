import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from zipfile import ZipFile

import tensorflow as tf
from tensorflow import keras
from keras import layers

from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

data_path = r'[removed]'
extract_path = r'[removed]'

if not os.path.exists(os.path.join(extract_path, "lung_colon_image_set")):
    with ZipFile(data_path, 'r') as zipf:
        zipf.extractall(extract_path)
        print('The data set has been extracted.')
else:
    print("Dataset já parece extraído. Pulando extração.")

path = r'[removed]'

classes = sorted(os.listdir(path))
print("Classes:", classes)

for cat in classes:
    image_dir = os.path.join(path, cat)
    images = os.listdir(image_dir)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Images for {cat} category', fontsize=18)

    for i in range(3):
        k = np.random.randint(0, len(images))
        img = np.array(Image.open(os.path.join(image_dir, images[k])))
        ax[i].imshow(img)
        ax[i].axis('off')
    plt.show()

IMG_SIZE = 224
SPLIT = 0.2
EPOCHS = 10
BATCH_SIZE = 8  # se travar, use 4

train_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    labels="inferred",
    label_mode="categorical",
    validation_split=SPLIT,
    subset="training",
    seed=2022,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    labels="inferred",
    label_mode="categorical",
    validation_split=SPLIT,
    subset="validation",
    seed=2022,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("class_names:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

try:
    train_ds = train_ds.cache().shuffle(300).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
except Exception:
    train_ds = train_ds.cache("train_cache").shuffle(300).prefetch(AUTOTUNE)
    val_ds = val_ds.cache("val_cache").prefetch(AUTOTUNE)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # congela o backbone (feature extractor)

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomFlip("vertical")(x)
x = layers.RandomRotation(0.08)(x)
x = layers.RandomZoom(0.10)(x)

x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=1,
    factor=0.5,
    min_lr=1e-5,
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[es, lr]
)

FINE_TUNE = True
if FINE_TUNE:
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        verbose=1,
        callbacks=[es, lr]
    )

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()

y_prob = model.predict(val_ds, verbose=0)
y_pred = np.argmax(y_prob, axis=1)
y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in val_ds], axis=0)

print(metrics.classification_report(y_true, y_pred, target_names=class_names))
