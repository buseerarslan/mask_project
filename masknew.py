# Kütüphane kurulumu
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

#  Veri yolları
train_dir = 'C:/Users/BUSE/Desktop/mask/data'

#  Veri yükleme ve augmentasyon
img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Class Weights hesaplama
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weight_dict)

#  EfficientNetB0 freeze ederek yükleme
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

#  Model oluşturma
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callback ekleme
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

# İlk eğitim (freeze edilmiş hali ile)
history = model.fit(
    train_data,
    epochs=15,
    validation_data=val_data,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

#  Fine-Tuning için son 20 katmanı açma
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Fine-tuning için düşük learning rate kullanılır
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#  Fine-Tuning eğitimi
history_finetune = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# Eğitim Sonrası Loss ve Accuracy grafikleri
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss (Freeze)')
plt.plot(history.history['val_loss'], label='Val Loss (Freeze)')
plt.plot(history_finetune.history['loss'], label='Train Loss (FT)')
plt.plot(history_finetune.history['val_loss'], label='Val Loss (FT)')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc (Freeze)')
plt.plot(history.history['val_accuracy'], label='Val Acc (Freeze)')
plt.plot(history_finetune.history['accuracy'], label='Train Acc (FT)')
plt.plot(history_finetune.history['val_accuracy'], label='Val Acc (FT)')
plt.legend()
plt.title('Accuracy over Epochs')

plt.show()

#  Confusion Matrix
val_data.reset()
pred = model.predict(val_data, verbose=1)
pred_labels = (pred > 0.5).astype(int)
true_labels = val_data.classes

cm = confusion_matrix(true_labels, pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:\n", classification_report(true_labels, pred_labels, target_names=['With Mask', 'Without Mask']))

#  5 örnek görsel üzerinde tahmin
import random

plt.figure(figsize=(15,10))
for i in range(5):
    idx = random.randint(0, len(val_data.filenames)-1)
    img_path = val_data.filepaths[idx]
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array_exp)[0][0]
    
    label = "Without Mask" if pred > 0.5 else "With Mask"
    color = 'green' if (label == "With Mask" and true_labels[idx]==0) or (label=="Without Mask" and true_labels[idx]==1) else 'red'

    plt.subplot(1,5,i+1)
    plt.imshow(img.astype(np.uint8))
    plt.title(f"Pred: {label}\n{'Correct' if color=='green' else 'Wrong'}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()
