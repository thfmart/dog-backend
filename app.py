import io
import traceback

from flask import Flask, request, g
from flask import send_file
from flask_mako import MakoTemplates, render_template
from plim import preprocessor

from PIL import Image, ExifTags
#from scipy.misc import imresize
import numpy as np
import keras
import tensorflow as tf

app = Flask(__name__, instance_relative_config=True)
# For Plim templates
mako = MakoTemplates(app)
app.config['MAKO_PREPROCESSOR'] = preprocessor
# app.config.from_object('config.ProductionConfig')


# Preload our model
model = keras.models.load_model('keras_model.h5')
graph = tf.get_default_graph()


def ml_predict(image):
    with graph.as_default():
        # Add a dimension for the batch
        #prediction = model.predict(image)
        prediction = model.predict(np.expand_dims(image, axis=0))
    return prediction


def decode_img(file_path):
    #img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [299, 299])
    img = img/255
    return img


def predict_breed(img_path):
    # load the image and return the predicted breed
    img = decode_img(img_path)
    print('decode')
    prediction = ml_predict(image)
    print('prediction sucessfull')
    prediction = prediction.argmax(axis=1)
    return prediction


def rotate_by_exif(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        if not orientation in exif:
            return image

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
        return image
    except:
        traceback.print_exc()
        return image


THRESHOLD = 0.5
@app.route('/predict', methods=['POST'])
def predict():
    # Load image
    image = request.files['file']
    print('request file sucessfull')
    image = Image.open(image)
    image = rotate_by_exif(image)
    print('rotate sucessfull')
    result = predict_breed(image)

    # Model input shape = (224,224,3)
    # [0:3] - Take only the first 3 RGB channels and drop ALPHA 4th channel in case this is a PNG
    #prediction = ml_predict(resized_image)
    #print('PREDICTION COUNT', (prediction[:, :, 1]>0.5).sum())
    #result = prediction.argmax(axis=-1)

    # Resize back to original image size
    # [:, :, 1] = Take predicted class 1 - currently in our model = Person class. Class 0 = Background
    #prediction = imresize(prediction[:, :, 1], (image.height, image.width))
    #prediction[prediction>THRESHOLD*255] = 255
    #prediction[prediction<THRESHOLD*255] = 0

    # Append transparency 4th channel to the 3 RGB image channels.
    #transparent_image = np.append(np.array(image)[:, :, 0:3], prediction[: , :, None], axis=-1)
    #transparent_image = Image.fromarray(transparent_image)

    # Send back the result image to the client
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)


@app.route('/')
def homepage():
    return render_template('index.html.slim', name='mako')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
