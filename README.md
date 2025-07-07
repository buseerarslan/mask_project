Face Mask Detection Proje Raporu
1- Kullanılan Model Mimarisi
Bu projede EfficientNetB0 mimarisi kullanılmıştır.

Yapı:
- EfficientNetB0, ImageNet ağırlıkları ile yüklenmiş, üst katmanları çıkarılmıştır.
- Model ağırlıkları freeze edilmiştir.

Üstüne eklenen katmanlar:
- GlobalAveragePooling2D
- Dense(128, relu)
- Dropout(0.3)
- Dense(1, sigmoid)

Eğitim:
- Loss fonksiyonu: binary_crossentropy
- Optimizasyon: Adam(learning_rate=0.0001)
- Metric: accuracy
- Epoch: 152) Eğitim Süreci ve Metrikler
- Veri seti eğitim (%80) ve doğrulama (%20) olarak ayrılmıştır.
- Görseller 224x224 boyutuna getirilmiş, 0-1 aralığında normalize edilmiştir.
- Basit augmentasyon uygulanmıştır (flip, zoom).
- Model 15 epoch boyunca frozen halde eğitilmiştir.

- Sonuç:

   
	                  precision	recall  	f1-score   	support
   With Mask       	0.00      	0.00      	0.00      	745
Without Mask       	0.51      	1.00      	0.67       	765
accuracy    			                          0.51      	1510
macro avg       	  0.25	      0.50	      0.34	      1510
weighted avg       	0.26	      0.51	      0.34	      1510

 ![image](https://github.com/user-attachments/assets/a4ab79e1-b597-4a5a-94c9-2824149aa744)

3-)Confusion Matrix ve Yorum
Confusion Matrix:
   Tahmin: Maske Var (0)   Tahmin: Maske Yok (1)  

Gerçek: Maske Var (0)	    0	    745
Gerçek: Maske Yok (1)    	0	    765
![image](https://github.com/user-attachments/assets/c862f5e0-6bc7-41cb-a25b-68c0f6b91a75)

 
Yorum:
- Model tüm örnekleri Maske Yok olarak sınıflandırmıştır.
- Maske takılı 745 görselin tamamı yanlış sınıflandırılmıştır.
- Maske olmayan 765 görselin tamamı doğru sınıflandırılmıştır.
- Bu durum, modelin tek bir sınıfa saplanarak çalıştığını ve veri dengesizliğini/öğrenme sorunu olduğunu göstermektedir.
4-) Görsel Tahmin Sonuçları
- Rastgele seçilen 5 görselde model tahminleri 'Maske Yok' olarak çıkmıştır.
- Görsellerin hepsi Maske Yok ise tahminler doğru gözükmüştür.
- Ancak bu, modelin gerçekte doğru öğrendiği anlamına gelmemektedir.
![image](https://github.com/user-attachments/assets/d2b885fb-3157-4eba-83d2-b03195a70484)

5-) Karşılaşılan Zorluklar ve Çözüm Önerileri
Zorluklar:
- Model tek sınıfa saplanmıştır.
- Veri dağılımı veya class weights ayarlanmamıştır.
- Freeze sonrası fine-tuning yapılmamıştır.
- Augmentasyon sınırlıdır.

Çözüm Önerileri:

 
  •  MobileNetV2 • ResNet50 modelleri de denenmiştir.
  •  Class weights eklenmeli.	
  •  Fine-tuning (son 20 katmanı açarak) yapılmalı.
  •  Augmentasyon artırılmalı.
  •  Validation üzerinde dikkatli metrik kontrolü yapılmalı.
  •  Veri yapısı, etiketler, klasör isimleri dikkatlice gözden geçirilmelidir.	


Model ilk haliyle temel bir pipeline sunmuştur. Ancak geliştirme adımları uygulanarak doğruluk, precision ve recall değerleri dengelenmelidir.
