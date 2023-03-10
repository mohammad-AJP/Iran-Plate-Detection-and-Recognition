import fileinput
import os
from pathlib import Path
from typing import Union
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import tensorflow 
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import ImageFont, ImageDraw, Image 
### to read persian strings 
import arabic_reshaper
from bidi.algorithm import get_display 
######################################################################################
def prepend_text(filename: Union[str, Path], text: str):
    with fileinput.input(filename, inplace=True) as file:
        for line in file:
            if file.isfirstline():
                print(text)
            print(line, end="")

for file in ['yolov7/models/common.py', 'yolov7/models/experimental.py', 'yolov7/models/yolo.py', 'yolov7/utils/datasets.py']:
    prepend_text(file, "import sys\nsys.path.insert(0, './yolov7')")


from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size
from yolov7.utils.torch_utils import select_device, TracedModel
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.plots import plot_one_box


weights = 'yolov7\weights\weights.pt'
device_id = 'cpu'
image_size = 640
trace = True
ocr = load_model('path\to\CNN_OCR.h5')
source_image_path = "yolov7\images_test\car2.jpg"
persian_char_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "الف", "ب", "د", "ق", "ه", "ج", "ل", "م", "ن", "پ", "ع", "توانیاب", "ص", "س", "ط", "ت", "و", "ی"]


# Initialize
device = select_device(device_id)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(image_size, s=stride)  # check img_size
source_image = cv2.imread(source_image_path)
######################################################################################################
if trace:
    model = TracedModel(model, device, image_size)

if half:
    model.half()  # to FP16
    
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

model.eval()
#########################################################################################################
## Using YOLO to extract plate coordinates from raw image
def detect_plate(source_image):
    # Padded resize
    img_size = 640
    stride = 32
    img = letterbox(source_image, img_size, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        
    with torch.no_grad():
        # Inference
        pred = model(img, augment=True)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=0, agnostic=True)

    plate_detections = []
    det_confidences = []
    
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], source_image.shape).round()

            # Return results
            for *xyxy, conf, cls in reversed(det):
                coords = [int(position) for position in (torch.tensor(xyxy).view(1, 4)).tolist()[0]]
                plate_detections.append(coords)
                det_confidences.append(conf.item())

    return plate_detections
###########################################################################################################
## using CNN_OCR_model to recognize persian characters in a plate
def get_class(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize the pixel values
    img_array = np.expand_dims(img_array, axis=0)  # add a batch dimension
    y_pred = ocr.predict(img_array)
    predicted_class = np.argmax(y_pred)
    return predicted_class
##########################################################################################################
## since python does not work well with persian characters, we need some modifications to receive a 
## character 
def get_persian(cls):
    char = persian_char_list[cls]
    char = arabic_reshaper.reshape(char)
    char = get_display(char)

    return char
##########################################################################################################
## applying CNN_OCR_Model to 8 parts of the plate and extract each char
def get_plate_char(img):
    plate_id = []
    ch1 = img[0:49, 16:31]
    ch_path = cv2.imwrite("ch1.jpg", ch1)
    cls = get_class("ch1.jpg")
    char1 = get_persian(cls)
    plate_id.append(char1)
    ch2 = img[0:49, 31:47]
    ch_path = cv2.imwrite("ch2.jpg", ch2)
    cls = get_class("ch2.jpg")
    char2 = get_persian(cls)
    plate_id.append(char2)
    ch3 = img[0:49, 47:74]
    ch_path = cv2.imwrite("ch3.jpg", ch3)
    cls = get_class("ch3.jpg")
    char3 = get_persian(cls)
    plate_id.append(char3)
    ch4 = img[0:49, 74:89]
    ch_path = cv2.imwrite("ch4.jpg", ch4)
    cls = get_class("ch4.jpg")
    char4 = get_persian(cls)
    plate_id.append(char4)
    ch5 = img[0:49, 89:103]
    ch_path = cv2.imwrite("ch5.jpg", ch5)
    cls = get_class("ch5.jpg")
    char5 = get_persian(cls)
    plate_id.append(char5)
    ch6 = img[0:49, 103:118]
    ch_path = cv2.imwrite("ch6.jpg", ch6)
    cls = get_class("ch6.jpg")
    char6 = get_persian(cls)
    plate_id.append(char6)
    ch7 = img[15:49, 120:134]
    ch_path = cv2.imwrite("ch7.jpg", ch7)
    cls = get_class("ch7.jpg")
    char7 = get_persian(cls)
    plate_id.append(char7)
    ch8 = img[15:49, 134:149]
    ch_path = cv2.imwrite("ch8.jpg", ch8)
    cls = get_class("ch8.jpg")
    char8 = get_persian(cls)
    plate_id.append(char8)
    return plate_id
##########################################################################################################
## Crop the plate, detect and recognize characters and show them in image. Then save image to the local drive
def plate_crop(plate_list):
    dim = (150, 50)
    c = np.array(plate_list)
    count = 0
    for i in c:
        x1 = i[0]
        y1 = i[1]
        x2 = i[2]
        y2 = i[3]
        cropped_img = source_image[y1:y2, x1:x2]
        #cv2.imshow("cropped", cropped_img)
        resized = cv2.resize(cropped_img, dim, interpolation = cv2.INTER_AREA)
        plate_number = get_plate_char(resized)
        plate_number = (' '.join(plate_number))
        ###Adding plate number to the source image
        # Load image in OpenCV  
        image = cv2.imread(source_image_path)  
        # Convert the image to RGB (OpenCV uses BGR)  
        cv2_im_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  
        # Pass the image to PIL  
        pil_im = Image.fromarray(cv2_im_rgb)  
        draw = ImageDraw.Draw(pil_im)  
        # use a truetype font  
        font = ImageFont.truetype("Yekan.ttf", 40)  
        # Draw the text  
        draw.text((x1,y1-30), plate_number, font=font)  
        # Get back the image to OpenCV  
        cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)  
        ###showing image for debugging purpose
        cv2.imshow('Fonts', cv2_im_processed)  
        cv2.waitKey(0)  
        cv2.destroyAllWindows()
        
        
    cv2.imwrite(source_image_path[19:] + ".jpg", cv2_im_processed)
   
##########################################################################################################

coord = detect_plate(source_image)
plate_crop(coord)
