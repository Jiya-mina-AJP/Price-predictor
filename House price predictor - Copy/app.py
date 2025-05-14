#!/usr/bin/env python
# coding: utf-8

# In[2]:


from flask import Flask, render_template, request
import pickle
import numpy as np


# In[3]:


app = Flask(__name__)


# In[4]:


pipe = pickle.load(open('RidgeModel.pkl', 'rb'))


# In[5]:


locations = [
    'Whitefield', 'Sarjapur  Road', 'Electronic City', 'Raja Rajeshwari Nagar',
    'Marathahalli', '7th Phase JP Nagar', 'Kanakpura Road', 'Uttarahalli',
    'Thanisandra', 'Hebbal', 'Bannerghatta Road', 'Yelahanka', 'Hennur Road',
    'Electronic City Phase II', 'Haralur Road', 'Rajaji Nagar', 'Bellandur',
    'KR Puram', 'Electronics City Phase 1', 'Hoodi', 'Harlur', 'Kalena Agrahara',
    'Horamavu', 'Begur Road', 'Kengeri', 'Gottigere', 'Varthur', 'Ramamurthy Nagar',
    'Kaggadasapura', 'Old Airport Road'
]


# In[7]:


@app.route('/')
def index():
    return render_template('index.html', locations=locations, prediction_text=None, error=None)



# In[8]:


@app.route('/predict', methods=['POST'])
def predict():
    import pandas as pd
    error = None
    prediction_text = None
    try:
        location = request.form.get('location')
        bhk = request.form.get('bhk')
        bath = request.form.get('bath')
        sqft = request.form.get('sqft')
        # Validate input
        if not (location and bhk and bath and sqft):
            raise ValueError("All fields are required.")
        bhk = int(bhk)
        bath = int(bath)
        sqft = float(sqft)
        input_data = pd.DataFrame([{
            'location': location,
            'bhk': bhk,
            'bath': bath,
            'total_sqft': sqft
        }])
        prediction = pipe.predict(input_data)[0]
        output = round(prediction, 2)
        prediction_text = f"Estimated Price: ₹{output} lakhs"
    except Exception as e:
        error = f"Error: {str(e)}"
    return render_template('index.html', prediction_text=prediction_text, locations=locations, error=error)


# In[9]:


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




