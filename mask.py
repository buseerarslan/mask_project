#  Kütüphane kurulumu
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Veri yolları
train_dir = 'C:/Users/BUSE/Desktop/mask/data'


#  Veri yükleme ve bölme
img_size = (224, 224) # Görselleri 224x224 boyutuna getir, 0-1 aralığında normalize et
batch_size = 32

# %20 validation ayrıldı
train_datagen = ImageDataGenerator(
    rescale=1./255, #  0-1 aralığında normalize  pikseller 0-255’ten 0-1 aralığına normalize ediliyor.                 
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

#  %80 eğitim verisi
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)
#  %20 doğrulama verisi
val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

#  EfficientNetB0 modeli yükleme ve freeze etme
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False #  Pretrained ağı freeze (dondur) 

# Model oluşturma
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy', #  Belirtilen loss
    metrics=['accuracy']    # Belirtilen metric
)

model.summary()
""""
GlobalAveragePooling2D → Conv katmanlarının çıktısını 1D vektöre indirir.

Dense(128, relu) → Öğrenilebilir katman.

Dropout(0.3) → Overfitting’i engellemek için %30 dropout.

Dense(1, sigmoid) → Çıkış katmanı, ikili sınıflandırma için sigmoid aktivasyon kullanır.

"""
# Model eğitimi
history = model.fit(
    train_data,
    epochs=15,
    validation_data=val_data
)

#  Loss ve Accuracy grafiklerini çizdirme
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.show()

#  Confusion Matrix ve Classification Report
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
"""
confusion_matrix → Gerçek ve tahmin etiketlerini kullanarak karışıklık matrisi oluşturur.

classification_report → Precision, recall, f1-score metriklerini hesaplayıp ekrana yazdırır

"""

#  5 görsel üzerinde tahmin ve yorumlama
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
    plt.imshow(img)
    plt.title(f"Pred: {label}\n{'Correct' if color=='green' else 'Wrong'}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()
