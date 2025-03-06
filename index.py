from flask import Flask,jsonify,request,render_template
from tensorflow.keras.models import load_model 
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import pickle
import tensorflow as tf
app=Flask(__name__)
import pickle
model = load_model("model.h5")
@app.route("/",methods=["GET"])
def Main():
    return render_template("./index.html")

@app.route("/",methods=["POST"])
def MainPost():
    data = request.get_json() 
    Text_Img=data["image"]
    base64_data = Text_Img.split(",")[1] if "," in Text_Img else Text_Img
    image_bytes = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_bytes))
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0) 
    prediction = model.predict(image)
    predicted_class = 1 if prediction[0][0] > 0.5 else 0

    return jsonify({"prediction": predicted_class, "message": "Real" if predicted_class == 1 else "Not Real"})

    

if __name__=="__main__":
    app.run(debug=True)