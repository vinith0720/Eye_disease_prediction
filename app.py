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