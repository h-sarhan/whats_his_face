# whats_his_face
 Neural Network based face recognition with the facenet algorithm
 
[![Generic badge](https://img.shields.io/pypi/pyversions/Django)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Version-1.0.0-green.svg)](https://shields.io/)

download the keras model here: [url to download](https://drive.google.com/open?id=1PZ_6Zsy1Vb0s0JmjEmVd8FS99zoMCiN1)


1.  Create a python environment inside the project folder

        python3 -m venv venv

2.  Activate the new environment from project root

        source venv/bin/activate    (for linux)
        \venv\Scripts\activate.bat  (for windows)

3.  Install the Requirements

        pip install -r requirements.txt

4.  Place the dataset images in custom folders inside another folder called train_imgs

        ./train_imgs/{name_here}/{a_min_of_5_images_recommended}
        
5.  Train the model by using this command

        python train.py

6.  Run the model by using this command

        python main.py
# License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the [MIT License](https://github.com/SiddhantNair/whats_his_face/blob/master/LICENSE) for public use.
