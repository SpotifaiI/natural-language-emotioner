from processor import Processor
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
processor = Processor()

classificator = processor.train()

class AskRequest(BaseModel):
    comment: str

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.post("/ask")
async def ask(data: AskRequest):
    comment = data.comment

    if not comment:
        return {
            "success": False,
            "error": "No comment"
        }

    tokens = processor.convert_word(comment)
    emotion = classificator.classify(tokens)

    processor.examples.append((tokens, emotion))

    return {
        "success": True,
        "emotion": emotion,
        "comment": comment
    }

@app.get("/stats")
async def stats():
    return {
        "success": True,
        "metrics": processor.stats(classificator)
    }
