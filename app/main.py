from flask import Flask, request, redirect, render_template
import os
app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "/mnt/c/Users/hasou/computer-science/Hackathon/whats_his_face/app/static/img/uploads"

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
  if request.method == "POST":
    if request.files:
      image = request.files["image"]
      image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
      print("Image saved")
      return redirect(request.url)
  return render_template("home.html")

if __name__ == "__main__":
  app.run(debug=True)