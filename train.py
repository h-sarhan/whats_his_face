from facenet_pytorch import InceptionResnetV1
import tensorflow as tf
import numpy as np
from mtcnn import MTCNN
import torch
import cv2
import os

names = []
embeddings = []
names_labels = []
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
models = tf.keras.models
layers = tf.keras.layers
detector = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

print('loading images from database')
for folder in os.listdir('train_imgs'):
    filepath = 'train_imgs/' + folder
    if os.path.isdir(filepath):
        names_labels.append(folder)
        for file_item in os.listdir(filepath):
            if not os.path.isdir(filepath + '/' + file_item):
                image = cv2.imread(filepath + '/' + file_item)
                if not image is None:
                    faces = detector.detect_faces(image)
                    if len(faces) == 0:
                        continue
                    x, y, w, h =  faces[0]['box']
                    face = image[y:y + h, x:x + w]
                    if face.size > 0:
                        face = cv2.resize(face, (160, 160))
                        face = face.transpose(2, 0, 1).astype(np.float32)

                        mean, std = face.mean(), face.std()
                        face = (face - mean) / std
                        face = torch.from_numpy(np.expand_dims(face, axis=0))

                        embedding = resnet(face).cpu().detach().numpy()
                        embeddings.append(embedding)
                        names.append([folder])

print('training image face classifier')
print(names_labels)
embeddings = np.asarray(embeddings)
names = np.asarray(names)
for i in range(len(names)):
    for j in range(len(names[i])):
        names[i][j] = names_labels.index(names[i][j])

names = names.astype(np.int64)
model = models.Sequential()
model.add(layers.Dense(512, input_shape=(1, 512), activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(len(names_labels), activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(embeddings, names, epochs=50)
model.save('model.h5')