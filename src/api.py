from fastapi import APIRouter
from pydantic import BaseModel
from src.rag_pipeline import ask_question

router = APIRouter()
class QuestionRequest(BaseModel):
    question: str

@router.post("/ask")
def ask_question_endpoint(request: QuestionRequest):
    """
    Endpoint to ask a question and get an answer.
    """
    print(f"Received question: {request.question}")
    try:
        answer = ask_question(request.question)
        return {"answer": answer}
    except Exception as e:
        raise Exception(f"Error processing question: {e}")
