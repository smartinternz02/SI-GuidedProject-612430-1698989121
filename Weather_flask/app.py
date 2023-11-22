

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

model = load_model('weather.h5',compile=False)

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to handle the image upload and make predictions
@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
       f = request.files['image']
       print("current path")
       basepath = os.path.dirname(__file__)
       print("current path", basepath)
       filepath = os.path.join(basepath,'uploads',f.filename)
       print("upload folder is ", filepath)
       f.save(filepath)
        
        
    img = Image.open(filepath)
    processed_img = preprocess_image(img)
    pred = model.predict(processed_img)
    dict={'cloudy': 0, 'foggy': 1, 'rainy': 2, 'shine': 3, 'sunrise': 4}
    pred_class=np.argmax(pred,axis=1)
    key_with_value = [key for key,value in dict.items() if value ==pred_class[0] ]
    weather = "Given image comes under " + str(key_with_value[0]+" weather classification")
    return jsonify({'weather': weather})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,debug = False, threaded = False)
