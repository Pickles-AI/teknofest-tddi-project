from pydantic import BaseModel
from typing import List

class TextAnalysisRequest(BaseModel):
    text: str

class AnalysisResult(BaseModel):
    id: int
    type: str
    entity: str
    related_sentence: str

class TextAnalysisResponse(BaseModel):
    results: List[AnalysisResult]