import json
import re

import cv2
# from pandas import np
from pytesseract import pytesseract, Output

# custom configurations
language = 'eng'
custom_config = r'--oem 1 --psm 6 -c preserve_interword_spaces=1'

# column headers
keywords = ['description', 'test', 'test name', 'unit', 'units', 'ref. range',
            'reference value', 'reference range', 'reference', 'result']

# find column headers
testNameColumn = ['description', 'test', 'test name']
unitColumn = ['unit', 'units']
rangeColumn = ['ref. range', 'reference value', 'reference range', 'reference']
resultColumn = ['result']

titleArray = []


# return image as text strings
def string_output(temp):
    # extract text
    text = pytesseract.image_to_string(temp, lang=language, config=custom_config)
    print(text)
    return text


# Returns result containing box boundaries, confidences
def data_output(temp):
    dict = pytesseract.image_to_data(temp, output_type=Output.DICT, config=custom_config)
    return dict


# identify column headers and data and return the result
def process_ocr(temp):
    # image to data and coordinates
    text = string_output(temp)
    dict = data_output(temp)

    testName, unit, refRange, result = '', '', '', ''
    # y value for column data (line with the results)
    name_y, result_y, range_y = 0, 0, 0

    x_testName, y_testName, w_testName, h_testName = 0, 0, 0, 0
    x_result, w_result = 0, 0
    x_unit, w_unit = 0, 0
    x_range, w_range = 0, 0

    ocr_array = {}

    # split the text into lines and loop over
    for line in text.splitlines():
        # skip the line if it is empty
        if len(line) == 0:
            continue

        # check if the line contains table headings
        lower = line.lower()
        count = sum([lower.count(word) for word in keywords])

        # if the table header is found, find x,y coordinates for each column name
        if count >= 3:
            print("Table headers", count)

            # spelling correction to identify headers
            # spell = Speller(lang='en')
            # line = spell(line)
            # print("****", line)

            # find x,y,w,h of column headers
            for i in range(len(dict['level'])):
                (x, y, w, h) = (dict['left'][i], dict['top'][i], dict['width'][i], dict['height'][i])

                # x values of column headers
                for col in line.lower().split():

                    if col == dict['text'][i].lower():
                        # draw boxes only around header
                        image = cv2.rectangle(temp, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        # cv2.imshow("Boxes", image)
                        # cv2.waitKey(0)

                        # # append words of the line to the array
                        # titleArray.append(col)
                        # print(titleArray)
                        #
                        # # find columns if more than 1 word is used for the header
                        # for j in range(len(titleArray)):
                        #     if j < 1:
                        #         continue
                        #
                        #     # if the concatenated word is a keyword
                        #     title = titleArray[j - 1] + " " + titleArray[j]
                        #     if title in testNameColumn:
                        #         print(title, "!!!!!!!!!!!!!")
                        #         w_testName = w

                        # check each array item to find col headers
                        # save x,w values of the column header
                        if col in testNameColumn:
                            print("header name", col)
                            x_testName, y_testName, w_testName, h_testName = x, y, w, h

                        if col in resultColumn:
                            print("header result", col)
                            x_result, w_result = x, w

                        if col in unitColumn:
                            print("header unit", col)
                            x_unit, w_unit = x, w

                        if col in rangeColumn:
                            print("header range", col)
                            x_range, w_range = x, w

                # execute only if  y values are greater than the column header
                if y > y_testName and dict['text'][i] != '':

                    # if x is in the range of the col header
                    if x < (x_testName + w_testName) or x < x_result:
                        testName = dict['text'][i]
                        name_y = y
                        print("data name", testName)

                    if (x_result - 5) < x < (x_result + w_result):
                        result = dict['text'][i]
                        result_y = y
                        print("data result", result)

                    if (x_unit - 5) < x < (x_unit + w_unit):
                        unit = dict['text'][i]
                        unit_y = y
                        print("data unit", unit)

                    if (x_range - 5) < x < (x_range + w_range):
                        refRange = dict['text'][i]
                        range_y = y
                        print("data range", refRange)

                    #  check if all data are taken from the same line by the y value
                    if (name_y + 4 <= result_y or name_y - 4 <= result_y) and (
                            range_y + 4 <= result_y or range_y - 4 <= result_y):
                        # break the loop if values are found
                        if (testName and result and refRange != '') or (testName and result and unit != ''):
                            # print(testName, result, unit, ":", refRange)
                            ocr_array = {"TEST NAME": testName, "RESULTS": result, "UNIT": unit, "RANGE": refRange}
                            break

                        # if testName and result and unit != '':
                        #     # print(testName, result, unit, ":", refRange)
                        #     ocr_array = {"TEST NAME": testName, "RESULTS": result, "UNIT": unit, "RANGE": refRange}
                        #     break

        # # if unable to locate the header, raise an error
        # else:
        #     return 'Error: Unable to locate header'
        #     # raise Exception("Sorry we are unable to process your report")

    print(ocr_array)
    return json.dumps(ocr_array)