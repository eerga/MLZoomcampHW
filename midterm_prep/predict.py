#!/usr/bin/env python
# coding: utf-8

import pickle
import uvicorn
from fastapi import FastAPI
import numpy as np
from typing import Dict, Any

app = FastAPI(title = "property-prediction")

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(re_property):
    result = pipeline.predict(re_property)
    
    restored_value = np.exp(result) - 1
    print("Predicted property value: ", round(float(restored_value.item()),2))
    return round(float(restored_value.item()),2)

app.post("/predict")
def predict(customer: Dict[str, Any]):
    prediction = predict_single(customer)

    return {
        "predicted_value": prediction
    }


# used the 10th value from the test data
property_data = {
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





