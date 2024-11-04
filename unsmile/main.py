from fastapi import FastAPI
from pydantic import BaseModel
from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer

app = FastAPI()

model_name = 'smilegate-ai/kor_unsmile'
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=-1,   
        return_all_scores=True,
        function_to_apply='sigmoid'
    )

class Comments(BaseModel):
    text: str

@app.post('/comments/')
async def texts(comment: Comments):
    for result in pipe(comment.text)[0]:
        if result['label'] == 'clean':
            clean_score = result['score']
            
            if clean_score >= 0.8:
                return {"msg": "Good Text"}
            else:
                return {"msg": "Bad Text"}
    
    return {"msg": "Label not found"} 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
