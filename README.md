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
