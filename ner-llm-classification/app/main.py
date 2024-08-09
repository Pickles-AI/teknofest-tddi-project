from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.ner_service import NERService
from app.services.llm_service import LLMService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

ner_service = NERService()
llm_service = LLMService()

class TextAnalysisRequest(BaseModel):
    text: str

class AnalysisResult(BaseModel):
    entity: str
    sentiment: str

class TextAnalysisResponse(BaseModel):
    entity_list: list[str]
    results: list[AnalysisResult]

@app.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    logger.info(f"Received text: {request.text}")

    try:
        ner_results = ner_service.analyze(request.text)
        logger.info(f"NER results: {ner_results}")

        results = []
        entity_list = []
        for item in ner_results:
            entity = item["word"]
            entity_list.append(entity)
            sentiment = llm_service.analyze_sentiment(request.text, entity)
            results.append(AnalysisResult(entity=entity, sentiment=sentiment))

        logger.info(f"Analysis results: {results}")

        return TextAnalysisResponse(
            entity_list=entity_list,
            results=results
        )
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)