# Flask Image Classification Application

This is a Flask application for predicting image classes using a pre-trained deep learning model. The app accepts an image upload, processes it, and predicts its class using a TensorFlow/Keras model.

## Features

- Upload an image to classify
- Predict the image class using a deep learning model
- Display the prediction result and uploaded image

## Setup Instructions

### Prerequisites

1. Python 3.12
2. Flask
3. TensorFlow
4. NumPy
5. A pre-trained model (`odir.h5`)

### Folder Structure

```
project-directory/
|-- static/
|   |-- predictions/
|-- templates/
|   |-- home.html
|   |-- about.html
|   |-- predict.html
|   |-- result.html
|-- app.py
|-- model/
|   |-- odir.h5
```

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd project-directory
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required packages:
   ```bash
   pip install Flask tensorflow numpy
   ```
4. Ensure the pre-trained model is placed in the `model/` directory.

### Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://127.0.0.1:5000`.

## Code

```python
from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
#from PIL import Image
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model("./model/odir.h5")

def predict_class():
    file = request.files['image']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'static', 'predictions', file.filename)
    print(file_path)
    file.save(file_path)
    img = image.load_img(file_path, target_size=(224,224))
    x = image.img_to_array(img) #image to array
    x = np.expand_dims(x,axis= 0) #changing the shape
    preds = model.predict(x)
    pred = np.argmax(preds, axis=1)
    index = ['cataract', 'diabetic retinopathy', 'glaucoma', 'normal']
    result = str(index[pred[0]]) 
    return {"img_path": file.filename, "pred_class": result}

# Routes

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        result = predict_class()
        return render_template('result.html', img_path=result['img_path'], result=result['pred_class'])

    return render_template('predict.html')

def main():
    app.run(debug=True)

if __name__ == '__main__':
    main()
```

## Templates

Ensure the following templates are created in the `templates/` directory:

### `home.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Home</title>
</head>
<body>
    <h1>Welcome to the Image Classifier</h1>
    <a href="/predict">Predict an Image</a>
    <a href="/about">About</a>
</body>
</html>
```

### `about.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About</title>
</head>
<body>
    <h1>About the Image Classifier</h1>
    <p>This application classifies medical images using a pre-trained deep learning model.</p>
</body>
</html>
```

### `predict.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predict</title>
</head>
<body>
    <h1>Upload an Image for Prediction</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Predict</button>
    </form>
</body>
</html>
```

### `result.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Result</title>
</head>
<body>
    <h1>Prediction Result</h1>
    <img src="/static/predictions/{{ img_path }}" alt="Uploaded Image">
    <p>Predicted Class: {{ result }}</p>
    <a href="/predict">Predict Another Image</a>
</body>
</html>
