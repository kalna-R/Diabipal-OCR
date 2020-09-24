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
from PIL import Image, ImageEnhance
from pytesseract import pytesseract, Output
from tabulate import tabulate

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# library to create a network
import networkx as nx

app = Flask(__name__)
cors = CORS(app)


#  ?url=
@app.route('/url', methods=['POST', 'GET'])
def processURL():
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
    test = 'test'

    x_name, x_flag, x_unit, x_range, x_result = 0, 0, 0, 0, 0
    y_name, y_flag, y_unit, y_range, y_result = 0, 0, 0, 0, 0

    testNameTableData = ''
    unitTableData = ''
    rangeTableData = ''
    resultTableData = ''
    flagTableData = ''

    # get the path to the image
    data = request.data
    datadict = json.loads(data)
    path = datadict['url']

    # send a GET request to the specified path & get the response in bytes
    response = requests.get(path)
    image_bytes = BytesIO(response.content)

    # open with PIL
    image_pil = Image.open(image_bytes)
    # image_pil.show()

    # load using opencv
    req = urllib.request.urlopen(path)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image_cv = cv2.imdecode(arr, -1)

    # rescale image
    def set_image_dpi(original_image):
        # set length/width
        length_x, width_y = original_image.size
        factor = max(1, int(IMAGE_SIZE / length_x))
        size = factor * length_x, factor * width_y
        # resize
        im_resized = original_image.resize(size, Image.LANCZOS)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        # set dpi to 300
        im_resized.save(temp_file.name, dpi=(300, 300))
        # im_resized.show()
        return temp_file.name

    # get grayscale image
    def get_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # enhance brightness
    def enhance_brightness(image_pil):
        enhance = ImageEnhance.Contrast(image_pil).enhance(1.5)
        # enhance.convert('RGB')
        opencv_enhanced = np.asarray(enhance)
        # opencv_enhanced = opencv_enhanced[:, :, ::-1].copy()
        gray_bright = get_grayscale(opencv_enhanced)
        return gray_bright

    enhanced_img = enhance_brightness(image_pil)

    # cv2.imshow("Enhanced cv2", enhanced_img)

    # grey should be passes to threshold
    def image_smoothening(img):
        ret1, th1 = cv2.threshold(img, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
        ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(th2, (1, 1), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th3

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

    def process_image_for_ocr(original_img):
        temp_filename = set_image_dpi(original_img)
        im_new = remove_noise_and_smooth(temp_filename)
        return im_new

    # processed image
    temp = process_image_for_ocr(image_pil)

    # extract text and write a text file
    text = pytesseract.image_to_string(temp, lang=language, config=custom_config)
    print(text)

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
            # print(count, " Column Headers")
            break

    # coordinates
    gray = cv2.cvtColor(np.float32(image_pil), cv2.COLOR_BGR2GRAY)
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
        # print("****", dict['text'][i])
        if len(dict['text'][i].lower()) >= 3:

            # x value of the test header
            if testName.lower().find(dict['text'][i].lower()) != -1 and (dict['text'][i] != ''):
                if testName.startswith('test'):
                    if dict['text'][i].lower() == 'test':
                        x_name = x
                        y_name = y

            # x value of the header
            if unit.lower().find(dict['text'][i].lower()) != -1 and (dict['text'][i] != ''):
                x_unit, y_unit = x, y

            # x value of the header
            if refRange.lower().find(dict['text'][i].lower()) != -1 and (dict['text'][i] != ''):
                if refRange.startswith('ref'):
                    if dict['text'][i].lower() == 'reference':
                        x_range = x
                        y_range = y

            if result.lower().find(dict['text'][i].lower()) != -1 and (dict['text'][i] != ''):
                # if result.startswith('result'):
                #     if dict['text'][i].lower() == 'result':
                        x_result = x
                        y_result = y

            if x_name == x or x_name == x + 2 or x_name == x - 1:
                testNameTableData = dict['text'][i]

            if x_result == x or x_result == x + 1 or x_result == x - 1:
                # print(dict['text'][i])
                resultTableData = dict['text'][i]

            if x_range == x or x_range == x + 1 or x_range == x - 1:
                rangeTableData = dict['text'][i]

            if x_unit == x or x_unit == x + 1 or x_unit == x - 1:
                unitTableData = dict['text'][i]

            # if x_flag == x or x_flag == x + 1 or x_flag == x - 1:
            #     flagTableData = dict['text'][i]

            # get each columns x coordinate
            # find values starting with that x coordinate

        ocr_array = {"TEST NAME": testNameTableData.lower(), "RESULTS": resultTableData.lower(), "UNIT": unitTableData,
                 "RANGE": rangeTableData.lower()}

    print("OCR output ", ocr_array)

    # *****************
    # funtion to read file
    def readFile(path):

        # read data from the excel file
        dataframe = pd.read_csv(path)
        df = dataframe.fillna('')

        return df.astype(str)

    # function to extract subject, relation and object for similar terms
    def processSynonyms(row):
        subj = ''
        objct = ''
        relation = ''
        triple = []

        if row['Synonym'] != '':
            subj = row['TestName']
            objct = row['Synonym']
            relation = 'similar'

            triple = [subj.lower(), relation, objct.lower()]
            return triple
        else:
            return;

    # To-do- spelling corretion survice/service
    # function to extract subject, relation and object for measurement unit
    def processUnits(row):
        subj = ''
        objct = ''
        relation = ''
        triple = []

        if row['Unit'] != '':
            subj = row['TestName']
            objct = row['Unit']
            relation = 'unit'

            triple = [subj.lower(), relation, objct]
            # print(triple)
            return triple

        else:
            return;

    # # function to extract subject, relation and object for range
    # def processRange(text):
    #
    #     subject = ''
    #     obj = ''
    #     relation = ''
    #     # holds the dependency tag of the previous token
    #     previous_token = ""
    #
    #     # iterate through every word in sentence
    #     for token in text:
    #
    #         # extract the subject
    #         if token.dep_ == "ROOT":
    #             # if previous dependency tag is null then the root is the subject
    #             if not previous_token:
    #                 subject = token.text
    #
    #         # extract the object
    #         # if a num is followed by a sym, save it temporarily as a object
    #         if token.pos_ == "SYM" or token.pos_ == "PUNCT":
    #             if previous_pos == "NUM":
    #                 obj = previous_text + "" + token.text
    #
    #         # if object!=null & obejct is followed by a sym, upadate the object variable
    #         if obj:
    #             if token.pos_ != "SYM":
    #                 obj = obj + token.text
    #
    #         # asssign the current dep & pos tag as the previous tokens
    #         previous_token = token.dep_
    #         previous_text = token.text
    #         previous_pos = token.pos_
    #     #         print(subject,"==>",obj)
    #
    #     relation = "in range"
    #
    #     if (obj.strip(), relation.strip(), subject.strip()):
    #         #         print(type(obj.strip()), "====")
    #         return [subject.strip(), relation.strip(), obj.strip()]
    #     else:
    #         return;

    # draw the knowledge graph
    def printGraph(triples):

        # initialize graph object
        G = nx.MultiDiGraph()

        # automatically create nodes if they don't exist when adding an edge
        for triple in triples:
            G.add_edge(triple[0], triple[2], relation=triple[1])

        node_color = [G.degree(v) for v in G]
        node_size = [1500 * G.degree(v) for v in G]

        # k = distance between edges
        pos = nx.spring_layout(G, k=10)

        nx.draw(G, pos, edge_color='black', node_size=node_size, node_color=node_color, alpha=0.9,
                cmap=plt.cm.Blues, labels={node: node for node in G.nodes()})
        nx.draw_networkx_edge_labels(G, pos, label_pos=0.5, font_size=10, font_color='k', font_family='sans-serif',
                                     font_weight='normal')
        # plt.axis('off')
        # plt.show()

        return G

    def hasNode(G, node):

        in_edges = G.in_edges(nbunch=node, data='relation', keys=True)

        out_edges = G.out_edges(nbunch=node, data='relation', keys=True)

        return in_edges, out_edges

    # corrections
    # data => output array of the ocr
    def wordCorrection(G, data):

        in_nodes, out_nodes = hasNode(G, data['TEST NAME'])

        if in_nodes:
            for u, v, keys, relation in in_nodes:
                # print("values for glucose", u, v)

                # units correction
                if relation == 'unit' and data['UNIT'] != v:
                    data['UNIT'] = v

                # range correction
                if relation == 'range' and data['RANGE'] != v:
                    data['RANGE'] = v

            return data

        if out_nodes:
            for u1, v1, keys, relation in out_nodes:
                # print("values for glucose", u1, v1)

                # units correction
                if relation == 'unit' and data['UNIT'] != v1:
                    data['UNIT'] = v1

                # range correction
                if relation == 'range' and data['RANGE'] != v1:
                    data['RANGE'] = v1

            return data

    triples = []

    # path to the file
    path = r'https://docs.google.com/spreadsheets/d/e/2PACX-1vRFOFNO-1FTpeJc-u0vHtzh8VrO7cg4M19Nxff82FCc-QAA1lTTtLFXyuWzmKvsrUbkCPKuMEjdfC27/pub?output=csv'

    # dataframe of data
    df = readFile(path)

    # process row by row to find triples
    for index, row in df.iterrows():

        # identify synonyms
        synonyms = processSynonyms(row)
        if synonyms:
            triples.append(synonyms)

        # identify units
        units = processUnits(row)
        if units:
            triples.append(units)

    graph = printGraph(triples)
    out_array = wordCorrection(graph, ocr_array)

    # *************
    final_array = json.dumps(out_array)
    print("Final array", final_array)
    return final_array


if __name__ == '__main__':
    app.run(debug="true")
