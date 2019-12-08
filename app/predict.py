from facenet_pytorch import InceptionResnetV1
import tensorflow as tf
from mtcnn import MTCNN
import numpy
import torch
import cv2
import os

detector = MTCNN()
model = tf.keras.models.load_model('models/model.h5')
resnet = InceptionResnetV1(pretrained='vggface2').eval()
# img = cv2.imread('train_imgs/siddhant_nair/IMG_0846.jpg')
names = ['ben_afflek', 'elton_john', 'hassan', 'jerry_seinfeld', 'madonna', 'mindy_kaling']
# for folder in os.listdir('../train_imgs'):
#     filepath = '../train_imgs/' + folder
#     if os.path.isdir(filepath):
#         names.append(folder)
def video_test():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        image = cv2.resize(frame, (640, 360))
        faces = detector.detect_faces(image)
        for result in faces:
            x, y, w, h = result['box']
            if (x > 0) & (y > 0) & (x + w < 640) & (y + h < 360):
                face = image[y:y + h, x:x + w]
                face = cv2.resize(face, (160, 160))
                face = face.transpose(2, 0, 1).astype(numpy.float32)

                mean, std = face.mean(), face.std()
                face = (face - mean) / std
                face = torch.from_numpy(numpy.expand_dims(face, axis=0))

                embedding = resnet(face).cpu().detach().numpy()
                prediction = names[numpy.argmax(model.predict([[embedding]]))]
                a = model.predict([[embedding]])
                num = max(a[0][0].tolist())
                font = cv2.FONT_HERSHEY_SIMPLEX
                if num > 0.98:
                    cv2.putText(image, f'{prediction} {round(num, 2)}%', (x, y - 10), font,
                                0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                else:
                    cv2.putText(image, f'Unknown {round(num, 2)}%', (x, y - 10), font,
                                0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.imshow('frame', image)
        if cv2.waitKey(16) & 0xFF == ord('q'):
            break

def image_predict(imageName):
    img = cv2.cvtColor(cv2.imread(f'./static/img/uploads/{imageName}'), cv2.COLOR_BGR2RGB)
    image = cv2.resize(img, (640, 360))
    faces = detector.detect_faces(image)
    print(faces)
    for result in faces:
        x, y, w, h = result['box']
        if (x > 0) & (y > 0) & (x + w < 640) & (y + h < 360):
            face = image[y:y + h, x:x + w]
            face = cv2.resize(face, (160, 160))
            face = face.transpose(2, 0, 1).astype(numpy.float32)

            mean, std = face.mean(), face.std()
            face = (face - mean) / std
            face = torch.from_numpy(numpy.expand_dims(face, axis=0))

            embedding = resnet(face).cpu().detach().numpy()
            prediction = names[numpy.argmax(model.predict([[embedding]]))]
            a = model.predict([[embedding]])
            num = max(a[0][0].tolist())
            font = cv2.FONT_HERSHEY_SIMPLEX
            if num > 0.98:
                cv2.putText(image, f'{prediction} {round(num, 2)}%', (x, y - 10), font,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            else:
                cv2.putText(image, f'Unknown {round(num, 2)}%', (x, y - 10), font,
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
    output_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'./static/img/predictions/{imageName}.jpg', output_img)
    return f'./static/img/predictions/{imageName}.jpg'


if __name__ == "__main__":
    video_test()
