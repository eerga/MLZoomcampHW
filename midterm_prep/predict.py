#!/usr/bin/env python
# coding: utf-8

import pickle
import uvicorn
from fastapi import FastAPI
import numpy as np
from typing import Dict, Any

from typing import Literal, Union
from pydantic import BaseModel, Field, conint, confloat

#request body definition
class Reproperty(BaseModel):
    land_use_description_rank: Union[Literal[1, 2, 3, 4], Literal[1.0, 2.0, 3.0, 4.0]]
    zipcode_4tier_rank: Union[Literal[1, 2, 3, 4], Literal[1.0, 2.0, 3.0, 4.0]]
    overall_condition_rank: Union[Literal[1, 2, 3, 4, 5, 6], Literal[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    heat_type_rank: Union[Literal[0, 1, 2, 3, 4, 5, 6], Literal[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    total_num_rooms: conint(ge=0, le=20)
    gross_area: confloat(ge=0)
    adjusted_tax: confloat(ge=0)
    bed_rms: Union[conint(ge=0), confloat(ge=0)]
    full_bth: Union[conint(ge=0), confloat(ge=0)]
    kitchens: Union[conint(ge=0), confloat(ge=0)]
    num_parking: Union[conint(), confloat()]
    floor_number: confloat()
    building_age: confloat()
    remodel_age: confloat()


# response
class PredictResponse(BaseModel):
    predicted_value: float



app = FastAPI(title = "property-prediction")

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(re_property):
    result = pipeline.predict(re_property)
    return result  
    
@app.post("/predict")
def predict(re_property: Reproperty) -> PredictResponse:
    result = predict_single(re_property.dict())
    restored_value = np.exp(result) - 1
    prediction = round(float(restored_value.item()),2)
    
    return PredictResponse(
        predicted_value=prediction)
    

# used the 10th value from the test data
re_property = {
    "land_use_description_rank": 2.00,
    "zipcode_4tier_rank": 3.00,
    "overall_condition_rank": 4.00,
    "heat_type_rank": 6.00,
    "total_num_rooms": 8.00,
    "gross_area": 3278.00,
    "adjusted_tax": 6152.92,
    "bed_rms": 4.00,
    "full_bth": 2.00,
    "kitchens": 1.00,
    "num_parking": 1.00,
    "floor_number": 1.50,
    "building_age": 97.00,
    "remodel_age": 6.00,
}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)





