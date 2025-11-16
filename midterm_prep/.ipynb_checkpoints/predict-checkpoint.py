#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pickle
import numpy as np


# In[32]:


with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


# In[33]:


def predict_single(re_property):
    result = pipeline.predict(re_property)
    
    restored_value = np.exp(result) - 1
    return round(float(restored_value.item()),2)


# In[34]:


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


# In[35]:


predict_single(property_data)


# In[ ]:




