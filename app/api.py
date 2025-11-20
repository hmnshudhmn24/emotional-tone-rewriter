from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import rewrite_tone

app=FastAPI()

class Req(BaseModel):
    sentence:str
    tone:str='friendly'

@app.post('/rewrite')
async def r(req:Req):
    return {"output":rewrite_tone(req.sentence,req.tone)}

@app.get('/')
async def root():
    return {"msg":"Tone Rewriter running"}
