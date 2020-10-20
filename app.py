import requests
from flask_cors import CORS

from flask import Flask, request

from image_processing import process_image_less_noise, process_image_for_ocr, get_image_cv, get_image_pil
from data_extraction import process_ocr

app = Flask(__name__)
cors = CORS(app)


#  ?url=
@app.route('/url', methods=['POST', 'GET'])
def processURL():

    # load image from the url
    image_cv = get_image_cv(request)

    # image processing applied image
    processed_image = process_image_less_noise(image_cv)

    # # returns text as string
    # text = string_output(processed_image)
    #
    # # returns a dictionary of box boundaries
    # boundary = data_output(processed_image)

    # extract text using ocr engine
    # returns an array of test name, result, unit, range
    result = process_ocr(processed_image)

    #  if the result array is empty apply more image processing techniques
    if result:
        # "http://127.0.0.1:3000/corrections"
        response = requests.post(url="https://diabipal-knowledge-graph.herokuapp.com/corrections", data=result)
        print(response.json())
        if response.json():
            return response.json()
        else:
            return 'error'
    else:
        # image processing applied image
        out_image = process_image_for_ocr(get_image_pil(request))

        # # returns text as string
        # text = string_output(out_image)
        #
        # # returns a dictionary of box boundaries
        # boundary = data_output(out_image)

        # extract text using ocr engine
        # returns an array of test name, result, unit, range
        out_array = process_ocr(out_image)

        if out_array:
            # "http://127.0.0.1:3000/corrections"
            response = requests.post(url="https://diabipal-knowledge-graph.herokuapp.com/corrections", data=out_array)
            print("Response", response.json())
            if response:
                return response.json()
            else:
                return 'error'
        else:
            return 'Unable to process your report!'


if __name__ == '__main__':
    app.run(debug="true")
