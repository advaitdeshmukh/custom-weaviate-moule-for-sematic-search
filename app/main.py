from fastapi import FastAPI, Response, status
# from vectorizer import Vectorizer, VectorInput
# from meta import Meta
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from pydantic import BaseModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Tokenize the input texts
async def convertToVec(input_texts):
    if isinstance(input_texts, str):
        input_texts = [input_texts]

    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embedding = embeddings[0].tolist()
    return embedding
    # print("e5: "+str(embedding))

class VectorInput(BaseModel):
    text: str


tokenizer = AutoTokenizer.from_pretrained('e5-large')
model = AutoModel.from_pretrained('e5-large')
app = FastAPI()

@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
def live_and_ready(response: Response):
    response.status_code = status.HTTP_204_NO_CONTENT


@app.get("/meta")
def meta():
    meta = {'name': 'e5-large', 'language': 'en'}
    return meta


@app.post("/vectors")
async def read_item(item: VectorInput, response: Response):
    try:
        # vector = await vec.vectorize(item.text)
        vector = await convertToVec(item.text)
        # print("Spacy Embeddings: "+str(vector.tolist()))
        return {"text": item.text, "vector": vector, "dim": len(vector)}
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": str(e)}