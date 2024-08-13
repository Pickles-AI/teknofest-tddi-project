Bu repo, çeşitli doğal dil işleme (NLP) modellerinin eğitim süreçlerini ve bu modellerin API olarak sunulmasını içeren kodları barındırmaktadır. Projede yer alan her bir dosya, belirli bir NLP görevi için özel olarak hazırlanmış olup, bu görevlerin nasıl gerçekleştirileceğine dair kapsamlı örnekler sunmaktadır.

![pickles_ai](https://github.com/user-attachments/assets/c9038fba-f6fa-49a5-882a-c77482a904da)

# İçerik

## 1. `main.py`

# NLP Pipeline FastAPI Uygulaması

Bu dosya, FastAPI kullanarak oluşturulmuş bir NLP (Doğal Dil İşleme) pipeline'ını içermektedir. Bu pipeline, Named Entity Recognition (NER), Aspect-Based Sentiment Analysis (ABSA) ve geleneksel duygu analizi modellerini birleştirerek kapsamlı bir metin analizi sunmaktadır.

### Genel Bakış

Bu FastAPI uygulaması, aşağıdaki NLP görevlerini gerçekleştiren bir API sunmaktadır:

1. Named Entity Recognition (NER)
2. Aspect-Based Sentiment Analysis (ABSA)
3. Geleneksel Duygu Analizi

Uygulama, verilen bir metin içerisindeki varlıkları (entities) tespit eder, bu varlıklar için duygu analizi yapar ve sonuçları JSON formatında döndürür.

### Gereksinimler

Proje için aşağıdaki kütüphaneler gerekmektedir:

- fastapi
- uvicorn
- torch
- transformers
- numpy
- pydantic

### Kurulum

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install fastapi uvicorn torch transformers numpy pydantic
   ```

2. Projeyi klonlayın:
   ```bash
   git clone [repo_url]
   cd [repo_directory]
   ```

3. Uygulamayı çalıştırın:
   ```bash
   python main.py
   ```

Uygulama varsayılan olarak `http://0.0.0.0:8000` adresinde çalışacaktır.

### Kullanılan Modeller

1. NER Modeli: "moarslan/bert-base-turkish-cased-ner"
2. ABSA Modeli: "yangheng/deberta-v3-base-absa-v1.1"
3. Duygu Analizi Modeli: "moarslan/bert-base-turkish-sentiment-analysis"

### Örnek İstek

```python
import requests

url = "http://localhost:8000/predict"
data = {"text": "Spotify'da müzik dinlerken hiçbir problem yaşamıyorum, ancak SoundCloud üzerinden yayınları dinlemek bazen kesintiye uğruyor ve bu durum oldukça sinir bozucu."}

response = requests.post(url, json=data)
print(response.json())
```
### Demo Video

![demo_video](https://github.com/user-attachments/assets/b5497126-70bd-4a04-85b7-ea5861c467d4)

## 2. `notebooks/turkish-bert-ner-fine-tune.ipynb`

Turkish BERT NER Fine-Tuning

Bu notebook, Türkçe metinler üzerinde Named Entity Recognition (NER) görevi için bir BERT modelini fine-tune etmeyi amaçlamaktadır. Proje, Hugging Face'in Transformers kütüphanesini kullanarak gerçekleştirilmiştir. Türkçe NER görevi için bir BERT modelini fine-tune etmek için adım adım bir süreç sunmaktadır. Proje şu ana adımları içerir:

1. Veri setinin yüklenmesi ve hazırlanması
2. BERT modelinin ve tokenizer'ın hazırlanması
3. Veri ön işleme ve etiketlerin hizalanması
4. Model eğitimi
5. Model değerlendirmesi
6. Eğitilmiş modelin kullanımı

### Gereksinimler

Proje için aşağıdaki kütüphaneler gerekmektedir:

- transformers
- datasets
- torch
- numpy
- seqeval
- pandas
- accelerate

### Veri Seti

Proje, [turkish-wikiNER](https://huggingface.co/datasets/turkish-nlp-suite/turkish-wikiNER) veri setini kullanmaktadır. Bu veri seti, Hugging Face Datasets kütüphanesi aracılığıyla yüklenmektedir.

### Model Eğitimi

Eğitim için [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased) modeli temel alınmıştır. Eğitim parametreleri şu şekildedir:

- Epoch sayısı: 4
- Batch boyutu: 16
- Öğrenme oranı: 2e-5
- Weight decay: 0.01

### Model Etiketleri

Bu NER modeli, toplam 19 farklı varlık türünü tanıyabilmektedir. Varlıklar, BIO (Beginning, Inside, Outside) formatını kullanmaktadır. Etiketler şunlardır:

1. O (Outside): Herhangi bir varlık içermeyen kelimeler için kullanılır.
2. B-* (Beginning): Bir varlığın başlangıcını işaret eder.
3. I-* (Inside): Bir varlığın devamını işaret eder.

Tanınan varlık türleri:

- CARDINAL: Sayısal değerler
- DATE: Tarihler
- EVENT: Olaylar
- FAC: Tesisler, binalar
- GPE: Ülkeler, şehirler, eyaletler
- LANGUAGE: Diller
- LAW: Yasalar, mevzuatlar
- LOC: Lokasyonlar
- MONEY: Para birimleri ve miktarları
- NORP: Milliyet, dini veya politik gruplar
- ORDINAL: Sıra sayıları
- ORG: Organizasyonlar
- PERCENT: Yüzde ifadeleri
- PERSON: Kişi isimleri
- PRODUCT: Ürünler
- QUANTITY: Miktarlar
- TIME: Zaman ifadeleri
- TITLE: Unvanlar
- WORK_OF_ART: Sanat eserleri, kitaplar, şarkılar

Her bir varlık türü için B- ve I- önekleri bulunmaktadır (örneğin, B-PERSON ve I-PERSON).

Bu geniş etiket yelpazesi, modelin çeşitli alanlarda ve farklı türdeki metinlerde kullanılabilmesini sağlamaktadır.

### Değerlendirme

Model performansı, F1 skoru kullanılarak değerlendirilmektedir. Değerlendirme için seqeval kütüphanesi kullanılmıştır.

### Erişim

Eğitilmiş model, Hugging Face Model Hub'a yüklenmiştir ve [moarslan/bert-base-turkish-cased-ner](https://huggingface.co/moarslan/bert-base-turkish-cased-ner) adresinden erişilebilir.

## 3. `notebooks/deberta-absa.ipynb`

### DeBERTa Modeli

DeBERTa (Decoding-enhanced BERT with Disentangled Attention) modeli, BERT'in geliştirilmiş bir versiyonudur. Bu model, iki temel yenilik sunar:

1. **Ayrıştırılmış Dikkat Mekanizması (Disentangled Attention)**: Bu mekanizma, kelimelerin içerik ve pozisyon bilgilerini ayrı ayrı kodlar. Bu sayede model, kelimelerin anlamlarını ve cümle içindeki konumlarını daha iyi anlayabilir.

2. **Geliştirilmiş Maske Kodlaması (Enhanced Mask Decoder)**: Bu özellik, modelin maskelenmiş kelimeleri tahmin ederken bağlamı daha iyi kullanmasını sağlar.

### ABSA (Aspect-Based Sentiment Analysis)

ABSA, bir metindeki belirli yönler (aspect) hakkındaki duygu durumunu analiz etmeye odaklanan bir NLP tekniğidir. Örneğin, bir restoran yorumunda yemek kalitesi, servis ve fiyat gibi farklı yönler için ayrı ayrı duygu analizi yapılabilir.

### DeBERTa ABSA Modeli

[yangheng/deberta-v3-base-absa-v1.1](https://huggingface.co/yangheng/deberta-v3-base-absa-v1.1) modeli, DeBERTa'nın güçlü özelliklerini ABSA görevine uyarlamıştır. Bu model:

1. Metindeki belirli yönleri (aspect) tanımlayabilir.
2. Her bir yön için ayrı ayrı duygu analizi yapabilir.
3. Bağlama duyarlı analizler gerçekleştirebilir.

### Tokenizer

DeBERTa ABSA modelinin tokenizer'ı, metni model için uygun girdi formatına dönüştürür. Bu tokenizer:

1. **Text Segmentation**: Metni kelimelere ve alt kelimelere böler.
2. **Special Tokens**: [CLS], [SEP] gibi özel tokenleri ekler. Bu tokenler, modele girdinin yapısı hakkında bilgi verir.
3. **Tokenize**: Her tokeni model tarafından anlaşılabilir sayısal bir ID'ye dönüştürür.
4. **Padding ve Truncation**: Girdileri sabit bir uzunluğa getirir.

### Kullanım Örneği

```python
inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
```

Bu kullanımda:
- `[CLS]`: Sınıflandırma tokeni, cümlenin başında bulunur.
- `sentence`: Analiz edilecek ana metin.
- `[SEP]`: Ayırıcı token, ana metin ile aspect arasında ve aspect'ten sonra kullanılır.
- `aspect`: Analiz edilecek spesifik yön.

Bu yapı, modele hangi cümlenin hangi yön için analiz edileceğini açıkça belirtir.

### Model Çıktısı

Model, her bir duygu sınıfı (olumsuz, nötr, olumlu) için bir olasılık değeri üretir. En yüksek olasılığa sahip sınıf, o aspect için tahmin edilen duygu durumunu temsil eder.

Bu detaylı yaklaşım, metin analizinde daha hassas ve bağlama duyarlı sonuçlar elde etmeyi sağlar.

## 4. `notebooks/classification-model.ipynb`

Türkçe Duygu Analizi Sınıflandırma Modeli

Bu notebook, Türkçe metinler üzerinde duygu analizi yapmak için BERT tabanlı bir sınıflandırma modelini içermektedir. Model, metinleri "Negatif", "Nötr" ve "Pozitif" olarak sınıflandırmaktadır.

Bu proje, `dbmdz/bert-base-turkish-cased` önce eğitilmiş modelini temel alarak Türkçe metinler için bir duygu analizi modeli geliştirmeyi amaçlamaktadır. Model, PyTorch ve Transformers kütüphaneleri kullanılarak eğitilmiştir.

### Gereksinimler

- Python 3.x
- PyTorch
- Transformers
- pandas
- numpy
- scikit-learn
- tqdm

### Model Eğitimi

Model eğitimi şu adımları içermektedir:

1. Veri hazırlama ve ön işleme
2. BERT modelini yükleme ve ince ayar için hazırlama
3. Eğitim ve doğrulama veri yükleyicilerinin oluşturulması
4. Model eğitimi (5 epoch)
5. En iyi modelin kaydedilmesi

Eğitim parametreleri:

- Model: dbmdz/bert-base-turkish-cased
- Epoch sayısı: 5
- Batch boyutu: 32
- Öğrenme oranı: 3e-5
- Weight decay: 3e-4
- Warmup oranı: 0.2

### Değerlendirme

Model performansı, aşağıdaki metrikler kullanılarak değerlendirilmiştir:

- F1 skoru
- Geri çağırma (Recall)
- Hassasiyet (Precision)
- Doğruluk (Accuracy)

Değerlendirme sonuçları için `eval_root()` fonksiyonu kullanılmıştır.

### Model ve Veri Seti Paylaşımı

- Model: [bert-base-turkish-sentiment-analysis](https://huggingface.co/moarslan/bert-base-turkish-sentiment-analysis)
- Veri Seti: [turkish-sentiment-analysis-dataset](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset)

## 5. `notebooks/llm-peft.ipynb`

Proje sırasında uyguladığımız PEFT (Parameter-Efficient Fine-Tuning) yöntemiyle LLM (Language Model) modeline ince ayar yapmaya çalıştık. Ancak, donanım yetersizliği nedeniyle tam anlamıyla uygulayamadık.

Few-shot öğrenme sonucunda elde edilen sonuçlar beklentilerin altında kaldı; belki daha fazla veri kullanarak daha iyi sonuçlar elde edebilirdik. Bu, modelin performansını artırmak için gelecekte dikkate alınması gereken önemli bir faktör olabilir.

## 6. `notebooks/llm-fine-tune.ipynb`

Trendyol/Trendyol-LLM-7b-chat-dpo-v1.0 modeli ile fine-tuning çalışmaları yapıldı, ancak beklenen sonuçlar elde edilemedi. Ayrıca, modelin inferans aşamasında farklı promptlarla denemeler yapıldı fakat sonuçlar tatmin edici değildi. Modelin mevcut konfigürasyonu ve veri seti ile uyumu yeterli düzeyde olmadığından, performans beklentilerimizi karşılamadı.

Ayrıca microsoft/Phi-3-mini-128k-instruct, TURKCELL/Turkcell-LLM-7b-v1 ve meta-llama/Meta-Llama-3.1-8B modellerini de prompt engineering ve örnek input-output çiftleriyle eğiterek test ettik. Fakat bu modellerden phi modeli çok küçük bir model olduğu için, turkcell ve llama-3 modelleri ise donanım yetersizlikleri nedeniyle istenilen sonuçları vermedi.

## 7. Diğerleri

Cümle çıkarımı ve analiz denemelerimizde nltk ve spaCy kütüphanelerini kullanarak sentence ve entity belirlemeye çalıştık. Ancak, bu denemeler beklentilerimizi karşılamadı. İşte bu süreçte karşılaştığımız bazı zorluklar ve gözlemlerimiz:

### Denemeler ve Gözlemler

#### Entity Tespiti ve Sentence Eşleştirme:

Entiteleri belirlemek için kullandığımız modeller, özellikle Türkçe metinlerde sentence sınırlarını doğru bir şekilde tanımlamakta zorlandı.
Sentence sınırlarının ve yapılarının karmaşıklığı nedeniyle, entiteler ile ilgili cümleleri doğru eşleştiremedik.

#### Sentence Extraction:

Metinleri cümlelere bölerek her bir sentence için entity tespiti yapmaya çalıştık.
Ancak, sentence'ların doğru bir şekilde ayrıştırılması ve ilgili entitelerin tespiti, mevcut araçların dil ve yapısal karmaşıklığı nedeniyle yeterince hassas olmadı.

### Yetersiz Sonuçlar:

Kullanılan yöntemlerle, entitelerin yalnızca extraction üzerinde çalıştık.
Sentence ve entity eşleştirmesi, elde edilen sonuçların beklenilen doğrulukta olmamasına neden oldu.
Türkçe dil yapısının karmaşıklığı, bu süreçte karşılaştığımız en büyük zorluklardan biri olarak öne çıktı.

### Sonuç

Bu denemeler, entity ve sentence tespiti konusundaki sınırlamaları ortaya koydu. Türkçe metinlerde daha iyi performans gösterebilecek özel eğitimli modellere veya daha ileri seviye dil işleme araçlarına ihtiyaç olduğunu fark ettik. Kullandığımız araçlar, sentence extraction ve entity tespiti için yeterince güçlü sonuçlar veremedi.

## 8. Lisans

Bu proje, MIT Lisansı altında lisanslanmıştır. Detaylı bilgi için [LICENSE](https://github.com/Pickles-AI/teknofest-tddi-project/blob/main/LICENSE.txt) dosyasını inceleyebilirsiniz.

---

Bu proje, Türkçe metinler üzerinde çeşitli NLP görevlerini gerçekleştirmek isteyenler için kapsamlı bir kaynak sunmaktadır. Modellerin eğitim süreçlerinden API servisi haline getirilmesine kadar tüm adımlar ayrıntılı olarak dokümante edilmiştir.
