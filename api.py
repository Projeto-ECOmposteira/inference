import os
import numpy as np 
import simplejson


from flask import Flask, request
from PIL import Image
from io import BytesIO

from model import generator, model

PORT: int = os.getenv('PORT', 8080)

app = Flask(__name__)

@app.route('/send_image', methods=['POST'])
def inference():

    response = {}

    for name, data in request.files.items():

        image_file = BytesIO(data.read())

        img = Image.open(image_file)
        img = img.resize((64, 64))

        data = np.array(img)
        data = np.expand_dims(data, 0)

        output = model(data)


        result = 'organic' if output < 0.5 else 'non-organic'

        response[name] = result

    return simplejson.dumps(response)

app.run('0.0.0.0', PORT)
