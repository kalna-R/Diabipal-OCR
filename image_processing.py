import tempfile
import urllib.request
from io import BytesIO
import json
import numpy as np

from flask import request
import requests

import cv2
from PIL import Image, ImageEnhance

IMAGE_SIZE = 1800
BINARY_THRESHOLD = 127


# load image using opencv
def get_image_cv(request):
    # get the path to the image
    data = request.data
    datadict = json.loads(data)
    path = datadict['url']

    # send a GET request to the specified path & get the response in bytes
    response = requests.get(path)
    image_bytes = BytesIO(response.content)

    # load using opencv
    req = urllib.request.urlopen(path)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image_cv = cv2.imdecode(arr, -1)

    return image_cv


# get image using PIL
def get_image_pil(request):
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

    return image_pil


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# function to execute if the image is less noisy
def process_image_less_noise(cv_image):
    gray = get_grayscale(cv_image)
    # cv2.imshow("gray", gray)

    ret1, th1 = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh1", th1)

    dest_not1 = cv2.bitwise_and(gray, th1, mask=None)

    # cv2.imshow('Processed Image ', dest_not1)
    # cv2.waitKey(0)

    return dest_not1


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


# enhance brightness
def enhance_brightness(image_pil):
    enhance = ImageEnhance.Contrast(image_pil).enhance(1.5)
    # enhance.convert('RGB')
    opencv_enhanced = np.asarray(enhance)
    # opencv_enhanced = opencv_enhanced[:, :, ::-1].copy()
    gray_bright = get_grayscale(opencv_enhanced)
    return gray_bright


# grey should be passes to threshold
# foreground pixels 255 and background 0
def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


# remove noise
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


# deskew image
def skew_correction(grey_img):
    gray = cv2.bitwise_not(grey_img)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = grey_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(grey_img, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # draw the correction angle on the image so we can validate it
    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the output image
    cv2.imshow("Input", grey_img)
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)

    return rotated


#  execute if the image is noisy
#  returns the processed image
def process_image_for_ocr(original_img):
    temp_filename = set_image_dpi(original_img)
    im_new = remove_noise_and_smooth(temp_filename)
    return im_new

