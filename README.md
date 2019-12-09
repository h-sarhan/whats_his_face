# whats_his_face

Neural network based face recognition web app.

Currently, the model is trained on a couple of celebrity faces (and me)

[![Generic badge](https://img.shields.io/pypi/pyversions/Django)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Version-1.0.0-green.svg)](https://shields.io/)



# Tech Stack

* Tensorflow
* OpenCV
* pyTorch
* Flask

# Try it out

I have deployed the app on [heroku](https://whats-his-face-v1.herokuapp.com/). 


# Steps to run app locally and contribute

1.  Create a python environment inside the project folder

        python3 -m venv venv

2.  Activate the new environment from project root

        source venv/bin/activate    (for linux)
        \venv\Scripts\activate.bat  (for windows)

3.  Install the Requirements

        pip install -r requirements.txt

4.  Start the flask server

        gunicorn application:app
    
5.  In your web browser go to:

        http://127.0.0.1:8000

6.  Upload pictures of Madonna, Ben Affleck, Elton John, Jerry Seinfeld, or Mindy Kaling and click submit

# Next Steps

1. Get deployment to work
2. Convert project to a Tensorflow.js/node app 
3. Train on larger dataset
4. Make a nicer UI
5. Add the ability to train on custom data from the user
6. Implement face recognition in real time using (don't even know if this is possible)



# License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the [MIT License](https://github.com/SiddhantNair/whats_his_face/blob/master/LICENSE) for public use.
