import cv2
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import image
from segment import char_segmentation

#mg = image.load_img(r'D:\Desktop\PBE\7.png', grayscale=True, target_size=(28, 28))

model = load_model('mnist.h5')
def predict(img_file_path):

    data = char_segmentation(img_file_path)
    pred = []
    for item in data:
        #x = np.array(img)
        #x = np.expand_dims(x, axis=2)
        x = item.reshape(1, 28, 28, 1)
        #images = np.vstack([x])

        prob = model.predict(x)
        pred.append(prob.argmax())
    return pred

#img_file_path = r"D:\Desktop\PBE\ChuVietTayPhieuDieuTra_Box.png"
#img_file_path = r'D:\Desktop\PBE\img20190110_0002-1.jpg'
pred = predict(img_file_path)
print(pred)