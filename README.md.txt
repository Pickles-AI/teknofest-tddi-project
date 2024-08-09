# Proje Hakkında

Bu depo, çeşitli doğal dil işleme (NLP) modellerinin eğitim süreçlerini ve bu modellerin API olarak sunulmasını içeren kodları barındırmaktadır. Projede yer alan her bir dosya, belirli bir NLP görevi için özel olarak hazırlanmış olup, bu görevlerin nasıl gerçekleştirileceğine dair kapsamlı örnekler sunmaktadır.

## İçerik

### 1. `main.py`
**FastAPI** kullanılarak oluşturulmuş olan bu dosya, API sunucusu olarak hizmet vermektedir. Bu dosya, [notebooks/deberta-absa.ipynb](notebooks/deberta-absa.ipynb) dosyasında geliştirilen **DeBERTa tabanlı Aspect-Based Sentiment Analysis (ABSA)** modelini bir servis haline getirmektedir. Bu sayede, ABSA modelinin sunduğu analizler API üzerinden kullanılabilir hale gelmiştir.

### 2. `notebooks/deberta-absa.ipynb`
Bu Jupyter Notebook, **Aspect-Based Sentiment Analysis** (ABSA) modelinin geliştirilmesi ve eğitimi için kullanılan kodları içermektedir. Model, DeBERTa tabanlı bir yapı üzerine inşa edilmiştir ve Türkçe metinler üzerinde duygu analizi yapma yeteneğine sahiptir. Bu dosya, modelin eğitim sürecini ve performansını optimize etmek için kullanılan tüm adımları detaylı bir şekilde sunmaktadır.

### 3. `notebooks/classification-model.ipynb`
Bu dosya, **Duygu Sınıflandırma Modeli**nin eğitimine dair kodları içermektedir. Model, Türkçe metinler üzerinde olumlu, olumsuz ve nötr duygu sınıflandırması yapmak için tasarlanmıştır. Eğitim süreci boyunca kullanılan veri ön işleme teknikleri ve model hiperparametre ayarları bu dosyada ayrıntılı olarak açıklanmaktadır.

#### **Model ve Veri Seti Bilgileri:**
- Model: [bert-base-turkish-cased-ner](https://huggingface.co/moarslan/bert-base-turkish-cased-ner)
- Veri Seti: [turkish-sentiment-analysis-dataset](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset)

### 4. `notebooks/turkish-bert-ner-fine-tune.ipynb`
Bu Jupyter Notebook, **Türkçe için NER (Named Entity Recognition)** modelinin eğitilmesi sürecini içermektedir. Model, BERT tabanlı bir yapı üzerine kurulmuş olup, Türkçe metinlerde isim varlıklarının tanınmasını sağlamaktadır. Bu dosyada, modelin nasıl eğitildiği ve test edildiği adım adım açıklanmıştır.

#### **Model ve Veri Seti Bilgileri:**
- Model: [bert-base-turkish-cased-ner](https://huggingface.co/moarslan/bert-base-turkish-cased-ner)
- Veri Seti: [turkish-wikiNER](https://huggingface.co/datasets/turkish-nlp-suite/turkish-wikiNER)

### 5. Diğer Dosyalar
Depoda yer alan diğer dosyalar, farklı NLP modellerinin geliştirilmesi sürecinde denenen yöntemleri ve kullanılan kodları içermektedir. Bu dosyalar, model optimizasyonu ve performans iyileştirmeleri için farklı yaklaşımlar sunmaktadır.

## Lisans

Bu proje, MIT Lisansı altında lisanslanmıştır. Detaylı bilgi için [LICENSE](LICENSE) dosyasını inceleyebilirsiniz.

---

Bu proje, Türkçe metinler üzerinde çeşitli NLP görevlerini gerçekleştirmek isteyenler için kapsamlı bir kaynak sunmaktadır. Modellerin eğitim süreçlerinden API servisi haline getirilmesine kadar tüm adımlar ayrıntılı olarak dokümante edilmiştir.
