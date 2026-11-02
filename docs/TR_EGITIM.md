# CardioWave Lab — Türkçe Eğitim Rehberi

## 1. Problem ve temel kavramlar
Bu proje gerçek hastalık tanısı koymaz. Matematiksel olarak üretilen üç farklı dalga ritmi örüntüsünü ayırt etmeyi öğretir. **Supervised classification** için her sinyalin etiketi önceden bilinir. Eğitim/validasyon/test ayrımı genelleme performansını ölçer.

## 2. Mimari
`data.py` sinyalleri üretir; `features.py` 14 boyutlu öznitelik vektörü çıkarır; `train.py` Random Forest eğitir, doğrulama ve test metriklerini üretir. `tests/` otomatik denetimlerdir. `.github/workflows/ci.yml` her push ve PR için testleri çalıştırır.

## 3. Sinyal modeli
Her yapay atım bir Gauss darbesiyle modellenir:
`g(t)=A exp(-(t-mu)^2/(2 sigma^2))`.
Farklı zaman aralıkları üç sentetik sınıfı oluşturur. Gerçek ECG fizyolojisini temsil etmez.

## 4. Özellik çıkarımı
Öznitelikler: ortalama, standart sapma, minimum/maksimum, RMS, ardışık fark, tepe sayısı, RR aralığı özetleri ve spektral entropi. RMS = `sqrt(mean(x^2))`. Spektral entropi = `-sum(p_i log p_i)`.

## 5. Random Forest
Birden çok karar ağacının oylamasıdır. Farklı ağaçlar farklı bootstrap örneklerinden öğrenerek varyansı azaltır. Bu projede sabit rastgele tohumla 200 ağaç kullanılır.

## 6. Değerlendirme
Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = 2PR/(P+R). Macro F1 üç sınıfın F1 ortalamasıdır. Confusion matrix hangi sınıfların karıştığını gösterir. Veri sızıntısını engellemek için aynı sentetik özne hem eğitimde hem testte bulunmaz.

## 7. Robustness deneyi
Test sinyallerine sabit tohumla ek Gauss gürültüsü eklenir. Temiz ve gürültülü Macro F1 karşılaştırılır. Bu, gerçek cihaz kaymalarını veya klinik güvenliği kanıtlamaz.

## 8. Kod deneyi önerileri
Ağaç sayısını 50/200/500 yapın; gürültü standart sapmasını 0.05/0.1/0.2 değiştirin; grup ayrımını kaldırarak veri sızıntısının metrikleri nasıl etkileyebileceğini karşılaştırın (yalnızca kontrollü deneyde).

## 9. Mülakat soruları ve cevapları
**Neden group split?** Aynı öznenin örneklerinin farklı bölümlere sızmasını önler.
**Macro F1 neden?** Sınıfları eşit ağırlıkla değerlendirir.
**Random Forest nasıl çalışır?** Rastgele örnekler/özellikler kullanan ağaçları birleştirir.
**Noise stress ne ölçer?** Yapay ek gürültü altında modelin duyarlılığını.
**Bu model klinikte kullanılabilir mi?** Hayır; veriler yapaydır ve klinik doğrulama yoktur.

## 10. Çalıştırma
`pip install -e ".[dev]"`; `pytest -q`; `python -m cardiowave --subjects 60 --output outputs/metrics.json`.
