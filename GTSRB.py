import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# 2) general config
data_dir   = r"F:\ML Intern Elevvo\CSV\GTSRB"  
IMG_SIZE   = (64, 64)
BATCH_SIZE = 64
SEED       = 1337
EPOCHS     = 16
AUTOTUNE   = tf.data.AUTOTUNE
class_names = [
    "Speed limit (20km/h)", 
    "Speed limit (30km/h)", 
    "Speed limit (50km/h)", 
    "Speed limit (60km/h)", 
    "Speed limit (70km/h)", 
    "Speed limit (80km/h)", 
    "End of speed limit (80km/h)", 
    "Speed limit (100km/h)", 
    "Speed limit (120km/h)", 
    "No passing", 
    "No passing for vehicles over 3.5 metric tons", 
    "Right-of-way at the next intersection", 
    "Priority road", 
    "Yield", 
    "Stop", 
    "No vehicles", 
    "Vehicles over 3.5 metric tons prohibited", 
    "No entry", 
    "General caution", 
    "Dangerous curve to the left", 
    "Dangerous curve to the right", 
    "Double curve", 
    "Bumpy road", 
    "Slippery road", 
    "Road narrows on the right", 
    "Road work", 
    "Traffic signals", 
    "Pedestrians", 
    "Children crossing", 
    "Bicycles crossing", 
    "Beware of ice/snow", 
    "Wild animals crossing", 
    "End of all speed and passing limits", 
    "Turn right ahead", 
    "Turn left ahead", 
    "Ahead only", 
    "Go straight or right", 
    "Go straight or left", 
    "Keep right", 
    "Keep left", 
    "Roundabout mandatory", 
    "End of no passing", 
    "End of no passing by vehicles over 3.5 metric tons"
]

np.random.seed(SEED)
tf.random.set_seed(SEED)

def read_csv_any(path):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    colmap = {c.lower(): c for c in df.columns}
    if "path" not in colmap:
        for cand in ["filename", "file", "image", "img", "img_path"]:
            if cand in colmap:
                df.rename(columns={colmap[cand]: "Path"}, inplace=True)
                break
    if "Path" not in df.columns and "path" in df.columns:
        df.rename(columns={"path": "Path"}, inplace=True)
    if "ClassId" not in df.columns:
        for cand in ["classid", "class_id", "label", "labels", "target", "class"]:
            if cand in df.columns:
                df.rename(columns={cand: "ClassId"}, inplace=True)
                break
    assert "Path" in df.columns, f"`Path` column missing in {path}."
    assert "ClassId" in df.columns, f"`ClassId` column missing in {path}."
    return df

train_csv = os.path.join(data_dir, "Train.csv")
test_csv  = os.path.join(data_dir, "Test.csv")

df_train = read_csv_any(train_csv)
df_test  = read_csv_any(test_csv)

train_paths = [os.path.normpath(os.path.join(data_dir, p)) for p in df_train["Path"].astype(str)]
test_paths  = [os.path.normpath(os.path.join(data_dir, p)) for p in df_test["Path"].astype(str)]

y_train_raw = df_train["ClassId"].astype(int).to_numpy()
y_test_raw  = df_test["ClassId"].astype(int).to_numpy()

def filter_missing(paths, labels):
    keep = [i for i,p in enumerate(paths) if os.path.exists(p)]
    return [paths[i] for i in keep], labels[keep]

train_paths, y_train_raw = filter_missing(train_paths, y_train_raw)
test_paths,  y_test_raw  = filter_missing(test_paths,  y_test_raw)

print("Train samples:", len(train_paths), " Test samples:", len(test_paths))

classes_sorted = sorted(np.unique(y_train_raw).tolist())
label2index = {lab: i for i, lab in enumerate(classes_sorted)}
index2label = {i: lab for lab, i in label2index.items()}

y_train = np.array([label2index[l] for l in y_train_raw], dtype=np.int32)
y_test  = np.array([label2index.get(l, -1) for l in y_test_raw], dtype=np.int32)

valid_test_idx = np.where(y_test >= 0)[0]
test_paths = [test_paths[i] for i in valid_test_idx]
y_test     = y_test[valid_test_idx]

num_classes = len(classes_sorted)
print("Num classes:", num_classes)
print("Label mapping (sample):", dict(list(label2index.items())[:5]))

train_paths_split, val_paths, y_train_split, y_val = train_test_split(
    train_paths, y_train, test_size=0.10, random_state=SEED, stratify=y_train
)


def decode_image(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)  
    return img, label

def make_ds(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode_image, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = make_ds(train_paths_split, y_train_split, shuffle=True)
val_ds   = make_ds(val_paths,        y_val,        shuffle=False)
test_ds  = make_ds(test_paths,       y_test,       shuffle=False)


data_aug = keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
    layers.RandomTranslation(0.05, 0.05),
], name="data_augmentation")

def build_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=43):
    inputs = keras.Input(shape=input_shape)

    
    x = layers.Rescaling(1./255)(inputs)
    x = data_aug(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="GTSRB_CNN")

model = build_cnn(num_classes=num_classes)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1),
    keras.callbacks.ModelCheckpoint("gtsrb_cnn.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
]

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)


test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"Test accuracy: {test_acc:.4f}")

# predictions for cm/report
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, cmap="Blues", fmt="g")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("Classification Report (indices):")
print(classification_report(y_true, y_pred, digits=4))

def preprocess_single_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)  
    img = tf.expand_dims(img, 0)   
    return img

import random

for i in range(9):
    idx = random.randint(0, len(val_paths) - 1)
    img_path = val_paths[idx]
    true_label = y_val[idx]

    img = preprocess_single_image(img_path)

    pred = model.predict(img, verbose=0)
    pred_label = np.argmax(pred)

    orig_img = plt.imread(img_path)

    plt.subplot(3, 3, i+1)
    plt.imshow(orig_img)
    plt.axis("off")

    mark = "✅" if pred_label == true_label else "❌"
    plt.title(f"Pred: {class_names[pred_label]}\nTrue: {class_names[true_label]} {mark}")

plt.tight_layout()
plt.show()

def preprocess_with_cv2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)  # (1, 64, 64, 3)
    return img

plt.figure(figsize=(10, 10))

for i in range(9):
    idx = random.randint(0, len(test_paths) - 1)
    img_path = test_paths[idx]
    true_label = y_test[idx]

    img_for_model = preprocess_with_cv2(img_path)

    pred = model.predict(img_for_model, verbose=0)
    pred_label = np.argmax(pred)

    img_display = cv2.imread(img_path)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    img_display = cv2.resize(img_display, IMG_SIZE)

    plt.subplot(3, 3, i+1)
    plt.imshow(img_display)
    plt.axis("off")
    plt.title(f"Pred: {class_names[pred_label]}\nTrue: {class_names[true_label]}")

plt.tight_layout()
plt.show()


train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_acc) + 1)

plt.figure(figsize=(10,5))

# Accuracy subplot
plt.subplot(1,2,1)
plt.plot(epochs, train_acc, label="Train Acc")
plt.plot(epochs, val_acc, label="Val Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.ylim(0.4,1)

# Loss subplot
plt.subplot(1,2,2)
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

# Adjust layout
plt.tight_layout(rect=[0,0.05,1,1])  

# Add test accuracy clearly under the figure
plt.figtext(0.5, 0.01, f"Final Test Accuracy: {test_acc*100:.2f}%", 
            ha="center", fontsize=14, fontweight="bold")

plt.show()