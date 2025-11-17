import requests

url = 'http://localhost:9696/predict'
#url = 'https://still-brook-1880.fly.dev/predict'

re_property = {
    "land_use_description_rank": 2.0,
    "zipcode_4tier_rank": 3.0,
    "overall_condition_rank": 4.0,
    "heat_type_rank": 6.0,
    "total_num_rooms": 8.0,
    "gross_area": 3278.0,
    "adjusted_tax": 6152.92,
    "bed_rms": 4.0,
    "full_bth": 2.0,
    "kitchens": 1.0,
    "num_parking": 1.0,
    "floor_number": 1.5,
    "building_age": 97.0,
    "remodel_age": 6.0
}

response = requests.post(url, json=re_property)

prediction = response.json()
print("price prediction is: ", prediction)