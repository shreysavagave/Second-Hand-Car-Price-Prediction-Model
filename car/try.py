import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
data = pd.read_csv('D:\shrey\machine learning\projects\car\cars_cleaned_data_new.csv')

# Load model and preprocessing steps
model = pickle.load(open('D:\shrey\machine learning\projects\car\Random_forest_regressor.pkl', 'rb'))


new_data_dict = {
    'year': [2014],
    'fuel': ['Diesel'],
    'car_running': ['less than 500000'],
    'mileage': ['more than 20'],
    'car': ['Maruti Swift Dzire']
}

# Convert dictionary to DataFrame
new_data_df = pd.DataFrame(new_data_dict)

ans = model.predict(new_data_df)

print(ans)