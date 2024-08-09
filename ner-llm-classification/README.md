# Metin Analiz API

Bu proje, NER (Named Entity Recognition) ve LLM (Large Language Model) teknolojilerini kullanarak metin analizi yapan bir API sunar. API, verilen metindeki önemli varlıkları (entities) tespit eder ve her bir varlık için metinden ilgili bir cümle veya ifade çıkarır.

## Özellikler

- Metin içindeki önemli varlıkları (kişi, kurum, yer adları vb.) tespit eder.
- Her varlık için metinden ilgili bir cümle veya ifade çıkarır.
- Çıkarılan ifadeler, varlık hakkında olumlu, olumsuz veya nötr bir anlam taşıyabilir.
- Parçalanmış token'ları (örn. "ro ##wen ##ta") otomatik olarak birleştirir.

## Gereksinimler

- Python 3.7+
- FastAPI
- Transformers
- PyTorch
- Uvicorn

## Kurulum

1. Projeyi klonlayın:

git clone https://github.com/sizin-kullanici-adiniz/metin-analiz-api.git
cd metin-analiz-api

2. Sanal bir ortam oluşturun ve aktive edin:

python -m venv venv
source venv/bin/activate  # Windows için: venv\Scripts\activate

3. Gerekli paketleri yükleyin:

pip install -r requirements.txt

## Çalıştırma

Uygulamayı geliştirme modunda çalıştırmak için:

uvicorn app.main:app --reload

API varsayılan olarak `http://localhost:8000` adresinde çalışacaktır.

## Kullanım

API'yi kullanmak için `/analyze` endpointine POST isteği gönderin:

curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text":"Turkcell diğer operatörlere göre daha kaliteli."}'

## API Yanıt Formatı

API, aşağıdaki formatta bir JSON yanıtı döndürür:

{
  "entity_list": ["Entity1", "Entity2", ...],
  "results": [
    {
      "entity": "Entity1",
      "comment": "Entity1 hakkında metinden çıkarılan ilgili cümle veya ifade"
    },
    {
      "entity": "Entity2",
      "comment": "Entity2 hakkında metinden çıkarılan ilgili cümle veya ifade"
    },
    ...
  ]
}

## Geliştirme

app/services/ner_service.py: NER işlemlerini gerçekleştiren servis.
app/services/llm_service.py: LLM işlemlerini gerçekleştiren servis.
app/main.py: FastAPI uygulamasının ana dosyası.