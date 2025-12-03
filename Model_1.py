import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import DenseNet121

# ---------------------------------------------------------
# 1. REPRODUCIBILITY
# ---------------------------------------------------------
np.random.seed(1337)
tf.random.set_seed(1337)

# ---------------------------------------------------------
# 2. DATASET LOADING (IMAGE DATASET FROM DIRECTORY)
# ---------------------------------------------------------

TRAIN_DIR = "C:/Users/Visha/OneDrive/Desktop/tamota/train"
VAL_DIR   = "C:/Users/Visha/OneDrive/Desktop/tamota/val"

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32
)

val_data = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32
)

# Normalize images
train_data = train_data.map(lambda x, y: (x / 255.0, y))
val_data   = val_data.map(lambda x, y: (x / 255.0, y))

# ---------------------------------------------------------
# 3. DISPLAY SAMPLE IMAGES
# ---------------------------------------------------------

def plot_sample_images(path, title_text):
    image_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    print(f"Showing samples from: {path}")

    fig, axs = plt.subplots(2, 3, figsize=(13, 9))

    for i in range(6):
        if i < len(image_files):
            img_path = os.path.join(path, image_files[i])
            img = mpimg.imread(img_path)
            label = image_files[i].split('.')[0]

            ax = axs[i // 3, i % 3]
            ax.imshow(img)
            ax.set_title(label)
            ax.axis('off')
        else:
            axs[i // 3, i % 3].axis('off')

    fig.suptitle(title_text, fontsize=16)
    plt.tight_layout()
    plt.show()


plot_sample_images(TRAIN_DIR + "/Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                   "Tomato Yellow Leaf Curl Virus")

plot_sample_images(TRAIN_DIR + "/Tomato___Bacterial_spot",
                   "Tomato Bacterial Spot")

# ---------------------------------------------------------
# 4. MODEL – DenseNet121 Transfer Learning
# ---------------------------------------------------------

conv_base = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(256, 256, 3),
    pooling='avg'
)

model = Sequential([
    conv_base,
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.35),
    BatchNormalization(),
    Dense(120, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------------------------------------
# 5. TRAINING THE MODEL
# ---------------------------------------------------------

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_data,
    epochs=5,
    validation_data=val_data,
    callbacks=[early_stop]
)

# ---------------------------------------------------------
# 6. EVALUATE MODEL
# ---------------------------------------------------------

loss, accuracy = model.evaluate(val_data)
print(f"\nValidation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# ---------------------------------------------------------
# 7. SAVE TRAINED MODEL
# ---------------------------------------------------------

model.save("tomato_leaf_disease_model.h5")
print("\nModel saved as: tomato_leaf_disease_model.h5")