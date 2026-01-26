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

data_path = r'removed'
extract_path = r'removed'

path = r'removed'

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

IMG_SIZE = 128
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

model = keras.models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),

    layers.Rescaling(1./255),

    layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),

    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get('val_accuracy', 0) > 0.90:
            print('\nValidation accuracy reached 90%, stopping training.')
            self.model.stop_training = True

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
    #callbacks=[es, lr, myCallback()]
    callbacks = [es]
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()

y_prob = model.predict(val_ds, verbose=0)
y_pred = np.argmax(y_prob, axis=1)
y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in val_ds], axis=0)

print(metrics.classification_report(y_true, y_pred, target_names=class_names))
