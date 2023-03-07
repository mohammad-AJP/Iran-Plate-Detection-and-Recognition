
# YOLOv7 for Iranian Plate Detection and Recognition

This repo explains the codes needed for finetuning a YOLOv7 on a plate-detection dataset and then use an OCR model to recognize the plate characters.

## Downloading dataset
To create the dataset for this project, I used two kaggle datasets, one including forign car plates and the other including Iranian car plates. To combine these two datasets, I used **Roboflow.com** which is a great tool to create your own dataset for object detection tasks. With this website, you can annotate images, or combine annotated datasets to create a new one, as I did here. 

The links to the datasets are:
https://www.kaggle.com/datasets/skhalili/iraniancarnumberplate

https://www.kaggle.com/datasets/andrewmvd/car-plate-detection

## Training a YOLOv7 for plate detection
To train a YOLOv7 on the dataset created, use the code **YOLOv7_Iran_Plate_detection.ipynb**. I used google colab to run this code. if the code does not work for you and you get the error bellow:

"RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)"

Try solving it by changing line 685 of **utils/loss.py** to:

from_which_layer.append((torch.ones(size=(len(b),)) * i).to('cuda'))

And adding the line below after line 756 of **fg_mask_inboxes**:

fg_mask_inboxes = fg_mask_inboxes.to(torch.device('cuda'))


## Train a CNN for Persian OCR
As usuall ocr methods do not work well on persian characters, we need to train our own ocr models to detect them. In this projects, I used a simple 2-layes CNN as the backbone of my OCR model for 10 epochs which achieved an accuracy of 97% on persian character recognition (feel free to use more powerful and complex models with more epochs for this purpose). To train the model, I used a dataset of segmented Iranian plates (each character is extracted as a single image). you can find this dataset by the link beloW:


https://www.kaggle.com/datasets/sajjadsoroori/iranis?select=M

70% of the data is dedicated for training and 20 and 10% of the data is dedicated for validation and test purposes, respectively by using the code **ocr_data_prepare.py**. 

## Plate Detection and Recognition
The main code for plate detection and recognition is **plate_detection_main.py**

The algorithm of the code is as below:

- Import necessary libraries. 

- Use codes mostly from **predict.py** to detect the coordinates of the plates bounding box. 

- Extract the plate as a separate image 

- Resize the extracted plate to a fixed size

- Separate each character of the plate

- Feed each image patch to the pretrained CNN_OCR_Model to recognize Persian characters (to do so, since python does not work well on Persian characters, I used two libraries: **arabic_reshaper** and **bidi** to work with persian chars)

- The recognized characters are then joined together and form a complete plate ID

- The plate ID is then shown on the image and the final image is saved to the local drive.


