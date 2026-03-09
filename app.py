from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import Annotated,Literal
import pickle

model = pickle.load(open("MLmodel/house_model.pkl","rb"))

app = FastAPI()

class UserInput(BaseModel):
    area : Annotated[int,Field(...,description="This is the area of the house in square meter")]
    bedrooms: Annotated[int,Field(...,description="Number of bedrooms in the house")]
    bathrooms: Annotated[int,Field(...,description="Number of bathrooms in the house")]
    stories: Annotated[int,Field(...,description="Number of floors in the house ")]
    mainroad: Annotated[Literal["yes","no"],Field(...,description="Is house present on the mainroad",examples=["yes","no"])]
    guestroom: Annotated[Literal["yes","no"],Field(...,description="Is there a guestroom??",examples=["yes","no"])]
    basement: Annotated[Literal["yes","no"],Field(...,description="Available basements??",examples=["yes","no"])]
    hotwaterheating: Annotated[Literal["yes","no"],Field(...,description="Hotwater available ??",examples=["yes","no"])]
    airconditioning: Annotated[Literal["yes","no"],Field(...,description="airconditioning available or not",examples=["yes","no"])]
    parking: Annotated[int,Field(...,description="Number of parking")]
    prefarea: Annotated[Literal["yes","no"],Field(...,description="prefarea",examples=["yes","no"])]
    furnishingstatus: Annotated[Literal["furnished","semi-furnished","unfurnished"],Field(...,description="Status of furnishing")]


@app.get('/')
def home():
    return {"message":"Welcome to house price prediction API"}

@app.get("/health")
def health():
    return{
        "status": "OK"
    }                                                  

@app.post("/predict")
def Prediction(data:UserInput):

    mainroad = 1 if data.mainroad == "yes" else 0
    guestroom = 1 if data.guestroom == "yes" else 0
    basement = 1 if data.basement == "yes" else 0
    hotwaterheating = 1 if data.hotwaterheating == "yes" else 0
    airconditioning = 1 if data.airconditioning == "yes" else 0
    prefarea = 1 if data.prefarea == "yes" else 0

    furnishing_map = {
        "furnished": 0,
        "semi-furnished": 1,
        "unfurnished": 2
    }

    features = [
        data.area,
        data.bedrooms,
        data.bathrooms,
        data.stories,
        mainroad,
        guestroom,
        basement,
        hotwaterheating,
        airconditioning,
        data.parking,
        prefarea,
        furnishing_map[data.furnishingstatus]
    ]

    prediction = model.predict([features])

    return{
         "predicted_price_rupees": float(prediction[0]),
         "predicted_price_lakh": round(prediction[0] / 100000, 2)
    }


