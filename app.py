from typing import Union

from fastapi import FastAPI
from fastapi.responses import RedirectResponse


app = FastAPI()


@app.get("/")
def read_root():
    return RedirectResponse(url='/docs')


@app.get("/v1/tokenize/")
def read_item(text: str):
    return {"tokenized_text": text}


@app.get("/v1/infer/")
def read_item(text: str, temperature: float):
    return {"infered_text": text, "temperature": temperature}


if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
