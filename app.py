import json

import requests
from flask_cors import CORS

from flask import Flask, request, jsonify, copy_current_request_context

from image_processing import process_image_less_noise, process_image_for_ocr, get_image_cv, get_image_pil
from data_extraction import process_ocr

# from rq import Queue
# from worker import conn

app = Flask(__name__)
cors = CORS(app)


#  ?url=
@app.route('/url', methods=['POST', 'GET'])
def processURL():
    # load image from the url
    image_cv = get_image_cv(request)

    # image processing applied image
    processed_image = process_image_less_noise(image_cv)

    # extract text using ocr engine
    # returns json string of test name, result, unit, range
    result = process_ocr(processed_image)

    # if result is None:
    # return jsonify({'Error': 'Unable to identify your report. Please try again'})

    # word correction using knowledge graph
    if result:
        print("Request directing to knowledge graph...")
        # response = requests.post(url="http://127.0.0.1:3000/corrections", data=result)
        response = requests.post(url="https://diabipal-knowledge-graph.herokuapp.com/corrections", data=result)
        print("Status from the KG API: ", response.status_code)

        # returns a json object
        if response.json():
            print("Response", response.json())
            return response.json()
        else:
            return {'Error'}

    #  if the result array is empty apply more image processing techniques
    else:
        # image processing applied image
        out_image = process_image_for_ocr(get_image_pil(request))

        # extract text using ocr engine
        # returns an array of test name, result, unit, range
        out_array = process_ocr(out_image)

        if out_array is None:
            return jsonify({'Error': 'Unable to identify your report. Please try again'})

        if out_array:
            print("Request directing to knowledge graph...")
            # response = requests.post(url="http://127.0.0.1:3000/corrections", data=result)
            response = requests.post(url="https://diabipal-knowledge-graph.herokuapp.com/corrections", data=out_array)
            print("Status code of the response KG API: ", response.status_code)

            if response.json():
                print("Response", response.json())
                return response.json()
            else:
                return {'Error'}
        else:
            return {'Error'}


# # create RQ queue
# q = Queue(connection=conn)
# # send jobs to Redis
# result = q.enqueue(processURL, result_ttl=80)
# print(result)

if __name__ == '__main__':
    app.run(debug="true")
