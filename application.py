from flask import Flask, request, redirect, render_template
import numpy as np
import tensorflow as tf
import cv2
from mtcnn import MTCNN
import os
from predict import image_predict

app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "./static/img/uploads"
no_img = 'no_img'
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
  if request.method == "POST":
    if request.files:
      image = request.files["image"]
      image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
      print(image.filename)
      print("Image saved")
      image_prediction = image_predict(image.filename)
      no_img = ''
      return render_template('home.html', image_prediction=f'{image_prediction}')
  return render_template("home.html")



if __name__ == "__main__":
  app.run(threaded=True)