import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Modeli yükle
model = tf.keras.models.load_model("model.keras")

# Tahmin yapılacak görsel yolu
image_path = "C:/Users/BUSE/Desktop/CNN/pixil-frame-6.png"

# Orijinal görseli yükle (renkli ya da gri fark etmez)
orig_img = tf.keras.preprocessing.image.load_img(image_path)

# Görseli 28x28 ve gri olarak modele uygun hale getir
img = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale", target_size=(28,28))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
input_img = np.expand_dims(img_array, axis=0)  # (1,28,28,1)

# Tahmin yap
preds = model.predict(input_img)[0]

# Tahmin edilen sınıf ve güven
pred_label = np.argmax(preds)
confidence = preds[pred_label]

print(f"Tahmin edilen sınıf: {pred_label}")
print(f"Güven skoru: {confidence:.4f}")

# Grafik çizimi
fig, axs = plt.subplots(1,3, figsize=(15,5))

# 1. Orijinal Görsel
axs[0].imshow(orig_img)
axs[0].set_title("Orijinal Görsel")
axs[0].axis("off")

# 2. 28x28 Gri Görsel
axs[1].imshow(img_array.squeeze(), cmap="gray")
axs[1].set_title("28x28 Gri Görsel")
axs[1].axis("off")

# 3. Tahmin Olasılıkları Bar Grafiği
axs[2].bar(range(10), preds, color='skyblue')
axs[2].set_xticks(range(10))
axs[2].set_xlabel("Rakamlar")
axs[2].set_ylabel("Güven Skoru")
axs[2].set_title("Tahmin Olasılıkları")
axs[2].set_ylim([0,1])

# Tahmin edilen sınıfı renklendir (barı kırmızı yap)
axs[2].bar(pred_label, preds[pred_label], color='red')

plt.tight_layout()
plt.show()
