import os
import zipfile
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ["KAGGLE_USERNAME"] = "REDACTED"
os.environ["KAGGLE_KEY"] = "REDACTED"

ZIP_NAME = "covid19-radiography-database.zip"
EXTRACT_DIR = "data"
DATASET_ROOT = os.path.join(EXTRACT_DIR, "COVID-19_Radiography_Dataset")

if not os.path.exists(ZIP_NAME):
    os.system("kaggle datasets download -d tawsifurrahman/covid19-radiography-database")

os.makedirs(EXTRACT_DIR, exist_ok=True)

if not os.path.exists(DATASET_ROOT):
    with zipfile.ZipFile(ZIP_NAME, "r") as z:
        z.extractall(EXTRACT_DIR)

print("Dataset root:", DATASET_ROOT)
print("Existe?", os.path.exists(DATASET_ROOT))

classes = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

patterns = [
    os.path.join(DATASET_ROOT, "COVID", "images", "*"),
    os.path.join(DATASET_ROOT, "Lung_Opacity", "images", "*"),
    os.path.join(DATASET_ROOT, "Normal", "images", "*"),
    os.path.join(DATASET_ROOT, "Viral Pneumonia", "images", "*"),
]

files, labels = [], []
for cls, pat in zip(classes, patterns):
    for f in glob.glob(pat):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            files.append(f)
            labels.append(cls)

df = pd.DataFrame({"filename": files, "class": labels})
print("Total imagens (filtradas em .../images/):", len(df))
print(df["class"].value_counts())

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
SEED = 42

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.25,
    horizontal_flip=True
)

def make_train_gen():
    return train_datagen.flow_from_dataframe(
        df,
        x_col="filename",
        y_col="class",
        classes=classes,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED
    )

def make_val_gen():
    return train_datagen.flow_from_dataframe(
        df,
        x_col="filename",
        y_col="class",
        classes=classes,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED
    )

train = make_train_gen()
val = make_val_gen()

print("train.class_indices:", train.class_indices)
print("val.class_indices:  ", val.class_indices)

num_classes = len(classes)

def plot_batch(generator, class_names, n=16, nrows=4, ncols=4, title="Batch (16 imagens)"):
    x, y = next(generator)
    n = min(n, len(x))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4, nrows * 4),
        dpi=120,
        constrained_layout=True
    )
    axes = np.ravel(axes)

    for i in range(nrows * ncols):
        ax = axes[i]
        if i < n:
            ax.imshow(x[i], cmap="gray")
            ax.set_title(class_names[np.argmax(y[i])], fontsize=11)
        ax.axis("off")

    fig.suptitle(title, fontsize=16)
    plt.show()

tmp_plot = make_train_gen()
plot_batch(tmp_plot, classes, title="Treino (preview) - somente RX de .../images/")

train = make_train_gen()
val = make_val_gen()

base = Xception(
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False
)
base.trainable = False

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    base,
    layers.Dropout(0.2),
    GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax"),
])

model.summary()

model.compile(
    optimizer=Adam(learning_rate=3e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

EPOCHS = 10

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", patience=1, factor=0.3),
]


reset_cb = tf.keras.callbacks.LambdaCallback(
    on_epoch_begin=lambda epoch, logs: (train.reset(), val.reset())
)

history = model.fit(
    train,
    epochs=EPOCHS,
    validation_data=val,
    callbacks=callbacks + [reset_cb]
)

MODEL_PATH = "xception_covid19_radiography_fast_cpu.keras"
model.save(MODEL_PATH)
print("Modelo salvo em:", MODEL_PATH)

plt.figure(figsize=(8, 5), dpi=120)
plt.plot(history.history.get("accuracy", []), label="train_acc")
plt.plot(history.history.get("val_accuracy", []), label="val_acc")
plt.title("Accuracy")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5), dpi=120)
plt.plot(history.history.get("loss", []), label="train_loss")
plt.plot(history.history.get("val_loss", []), label="val_loss")
plt.title("Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
