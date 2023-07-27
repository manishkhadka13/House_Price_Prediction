from fastapi import FastAPI,HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel
from typing import Any
import uvicorn

app=FastAPI()

class housefeatures(BaseModel):
    location: str
    total_sqft: float
    bath: int
    bhk: int

model=pickle.load(open("./models/prediction.pkl","rb"))

@app.post("/predict")
def predict_house_price(data:housefeatures):
    try:
        x_new=pd.DataFrame([data.dict()])
        prediction=model.predict(x_new)
        prediction=prediction[0]
        return {"prediction":prediction}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8084,reload=True)