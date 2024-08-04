import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
data = pd.read_csv('cars_cleaned_data_new.csv')

# Load model and preprocessing steps
model = pickle.load(open('LinearRegressor.pkl', 'rb'))


@app.route('/')
def index():
    car = sorted(data['car'].unique())
    mileage = sorted(data['mileage'].unique())
    year = sorted(data['year'].unique())
    fuel = sorted(data['fuel'].unique())
    car_running = sorted(data['car_running'].unique())

    return render_template('index.html', car=car, mileage=mileage, year=year, fuel=fuel, car_running=car_running)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        car = request.form.get('car')
        mileage = request.form.get('mileage')
        year = request.form.get('year')
        fuel = request.form.get('fuel')
        car_running = request.form.get('car_running')

        # Prepare data for preprocessing
        input_data = {
            'year': [int(year)],
            'fuel': [fuel],
            'car_running': [car_running],
            'mileage': [mileage],
            'car': [car],
        }
        # Create DataFrame
        new_data_df = pd.DataFrame(input_data)

        prediction = model.predict(new_data_df)
        prediction = abs(prediction)

        return render_template('index.html', prediction=f"The predicted selling price is: {prediction[0]:,.2f} INR",
                               car=car, mileage=mileage, year=year, fuel=fuel,
                               car_running=car_running)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
