import base64
import tempfile
import urllib.request
from io import BytesIO
from flask_cors import CORS
import json
from json import JSONEncoder

from flask import Flask, request, jsonify
import requests

import cv2
import numpy as np
from PIL import Image
from pytesseract import pytesseract, Output
from tabulate import tabulate

app = Flask(__name__)
cors = CORS(app)


#  ?url=
@app.route('/url', methods=['POST', 'GET'])
def processURL():
    # get path from url

    data = request.data
    datadict = json.loads(data)

    path = datadict['url']
    print(type(path))
    print(path)

    # url = request.args.get('url')
    # str1 = url.split('files/')
    # str2 = str1[0] + 'files%2F'
    # str3 = str1[1].split('?alt')
    # url1 = str2 + str3[0]

    # return a json object with image info from the 'url'
    # r = requests.get(url1).json()
    # token = r['downloadTokens']

    # complete path for the image in firebase storage
    # path = url1 + '?alt=media&token=' + token

    IMAGE_SIZE = 1800
    BINARY_THRESHOLD = 180

    # custom configurations
    # custom_psm_config = r'--psm 4'
    language = 'eng'
    custom_config = r'--oem 1 --psm 6 -c preserve_interword_spaces=1'

    # return {"path": path}
    #
    # find column headers
    testNameColumn = ['description', 'test', 'test name']
    unitColumn = ['unit', 'units']
    rangeColumn = ['ref. range', 'reference value', 'reference range']
    resultColumn = ['result']
    flagColumn = ['flag']

    count = 0
    testName = ''
    unit = ''
    refRange = ''
    result = ''
    flag = ''

    x_name, x_flag, x_unit, x_range, x_result = 0, 0, 0, 0, 0
    y_name, y_flag, y_unit, y_range, y_result = 0, 0, 0, 0, 0

    testNameTableData = ''
    unitTableData = ''
    rangeTableData = ''
    resultTableData = ''
    flagTableData = ''

    # define the path to the image
    # path = r'E:\Reserach\Resources\Dataset\FBS\Glucose-Test-Results.jpg'
    # load the image from the specified location/directory using openCV
    # image = cv2.imread(path)
    response = requests.get(path)
    image_bytes = BytesIO(response.content)

    img = Image.open(image_bytes)
    # img.show()

    # rescale image
    def set_image_dpi(file_path):
        im = Image.open(file_path)
        length_x, width_y = im.size
        factor = max(1, int(IMAGE_SIZE / length_x))
        size = factor * length_x, factor * width_y
        im_resized = im.resize(size, Image.ANTIALIAS)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_filename = temp_file.name
        im_resized.save(temp_filename, dpi=(300, 300))
        return temp_filename

    def image_smoothening(img):
        ret1, th1 = cv2.threshold(img, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
        ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(th2, (1, 1), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th3

    # def inverse(img):
    #     ret, tv1 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    #     return ret

    def remove_noise_and_smooth(file_path):
        # load the image in grayscale mode using 0
        img = cv2.imread(file_path, 0)
        filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,
                                         3)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        img = image_smoothening(img)
        or_image = cv2.bitwise_or(img, closing)
        return or_image

    def process_image_for_ocr(file_path):
        temp_filename = set_image_dpi(file_path)
        im_new = remove_noise_and_smooth(temp_filename)
        return im_new

    # processed image
    # temp = process_image_for_ocr(path)

    # extract text and write a text file
    text = pytesseract.image_to_string(img, lang=language, config=custom_config)
    print("final text", text)
    # return text

    for line in text.splitlines():
        for item in testNameColumn:
            nameExists = line.lower().find(item)
            if nameExists != -1:
                testName = item
                count = count + 1

        for item in unitColumn:
            unitExists = line.lower().find(item)
            if unitExists != -1:
                unit = item
                count = count + 1

        for item in rangeColumn:
            rangeExists = line.lower().find(item)
            if rangeExists != -1:
                refRange = item
                count = count + 1

        for item in resultColumn:
            resultExists = line.lower().find(item)
            if resultExists != -1:
                result = item
                count = count + 1

        for item in flagColumn:
            flagExists = line.lower().find(item)
            if flagExists != -1:
                flag = item
                count = count + 1

        if count == 4:
            print(count, " Column Headers")
            break

    # coordinates
    gray = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
    # Returns result containing box boundaries, confidences
    dict = pytesseract.image_to_data(gray, output_type=Output.DICT, config=custom_config)
    # dict = {'level': [1,1,2], 'text':['','']},
    n_boxes = len(dict['level'])

    # range => starting from 0 to n_boxes
    for i in range(n_boxes):
        (x, y, w, h) = (dict['left'][i], dict['top'][i], dict['width'][i], dict['height'][i])
        # starting point = (x,y)
        # ending point = (x + w, y + h)
        # color = (0, 0, 255)
        # thickness
        # image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if len(dict['text'][i].lower()) >= 3:

            if testName.lower().find(dict['text'][i].lower()) != -1 and (dict['text'][i] != ''):
                if testName.startswith('test'):
                    if dict['text'][i].lower() == 'test':
                        x_name = x
                        y_name = y
            if flag.lower().find(dict['text'][i].lower()) != -1 and (dict['text'][i] != ''):
                x_flag = x
                y_flag = y

            if unit.lower().find(dict['text'][i].lower()) != -1 and (dict['text'][i] != ''):
                x_unit, y_unit = x, y

            if refRange.lower().find(dict['text'][i].lower()) != -1 and (dict['text'][i] != ''):
                if refRange.startswith('ref'):
                    if dict['text'][i].lower() == 'reference':
                        x_range = x
                        y_range = y

            if result.lower().find(dict['text'][i].lower()) != -1 and (dict['text'][i] != ''):
                x_result = x
                y_result = y

            if x_name == x or x_name == x + 1 or x_name == x - 1:
                testNameTableData = dict['text'][i]

            if x_result == x or x_result == x + 1 or x_result == x - 1:
                resultTableData = dict['text'][i]

            if x_flag == x or x_flag == x + 1 or x_flag == x - 1:
                flagTableData = dict['text'][i]

            if x_range == x or x_range == x + 1 or x_range == x - 1:
                rangeTableData = dict['text'][i]

            # get each columns x coordinate
            # find values starting with that x coordinate

            # data for the table
            table = [[testNameTableData, resultTableData, flagTableData, unitTableData, rangeTableData]]

    # array
    array = {testName.upper(): testNameTableData, result.upper(): resultTableData, flag.upper(): flagTableData,
             unit.upper(): unitTableData, refRange.upper(): rangeTableData}
    print(array)

    return {"array": array}


if __name__ == '__main__':
    app.run(debug="true")
