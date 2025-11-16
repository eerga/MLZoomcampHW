#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np
import sklearn

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')

# Training on the final model

def load_data():
    data_url = 'cleaned_property_data.csv'
    df = pd.read_csv(data_url)
    return df


def train_model(df):
    numerical = ['total_num_rooms','gross_area', 'adjusted_tax', 'bed_rms', 'full_bth', 
                          'kitchens', 
                         'num_parking', 'floor_number', 
                          'building_age', 'remodel_age'] 

    categorical = [
                            'land_use_description_rank',
                            'zipcode_4tier_rank',
                            'overall_condition_rank',
                            'heat_type_rank'
                         ]



    pipeline = make_pipeline(
        DictVectorizer(),
        ElasticNet(alpha=0.01, l1_ratio=0.01, max_iter=2000, random_state=1)
    )
    
    target = 'total_value'

    
    df_linear = df[categorical + numerical + [target]]
    df_full_train, df_test = train_test_split(df_linear, test_size=0.2, random_state=1)
    re_test_value = df_test.iloc[10]
    
    y_train = np.log(df_linear.total_value.values + 1)
    df_full_train = df_full_train.reset_index(drop = True)
    
    train_dict = df[categorical + numerical].to_dict(orient='records')
    
    pipeline.fit(train_dict, y_train)
    return re_test_value, pipeline 


def save_model(filename, model):
    with open(filename, 'wb') as f_out:
        pickle.dump(model, f_out)
    print(f'model saved to {filename}')


df = load_data()
test_value, pipeline  = train_model(df)
save_model('model.bin', pipeline)





