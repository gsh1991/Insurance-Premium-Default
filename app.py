
import numpy as np
from flask import Flask, request, render_template
import pickle

app=Flask(__name__,template_folder='templates')
model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    # Add a constant feature with a value of 0
    float_features.append(0)
    final_features=[np.array(float_features)]
    prediction=model.predict(final_features)

    if prediction == 1:
        prediction_text = "Customer will not Default"
    else:
        prediction_text = "Customer will default"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)