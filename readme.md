# Standford Cars Classifier

This repository is created for Grab AI Challenge 2019 (Computer Vision Challenge).
This work reference on the following works.
1. Github mazenmel (https://github.com/mazenmel/Stanford-Car-Dataset)
2. Kaggle deepBear (https://www.kaggle.com/deepbear/pytorch-car-classifier-90-accuracy)
3. Github andrewjong (https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d)

## Many thanks for their hard work as this repository will not be possible without their work.

You should run the code in the following order.

1. Pre-processing.ipynb
2. Feature Engineering.ipynb
3. Grab AI Challenge.ipynb
4. Get Dominant Colour of Car.ipynb

The code will automatically create the required folder and files.



## Prerequisite

Please download the 2 datasets and devkit from the following link and make sure you put these 3 files in the same directory as the code in this repository.
https://ai.stanford.edu/~jkrause/cars/car_dataset.html

Please make sure your workstation had internet access as Grab AI Challenge.ipynb will download pretrained resnet101 model from pytorch.



## Library

This work required the following library.
Please make sure the following library is installed.

scipy
numpy
pandas
os
tarfile
shutil
torch
torchvision
time
opencv-python (cv2)
sklearn
webcolors

If you run the Grab AI Challenge.ipynb at your local workstation, please comment the following line from second cell.

dataset_dir = "../input/process-car2/process_car/"

Then, uncommment the following line from second cell.

dataset_dir = ""

This is because this code tested with Kaggle kernel with GPU due to the heavy computation in training process.



## Pretrained model

Grab AI Challenge.ipynb uses pretrained resnet101 model from pytorch.
You can manually download from this link https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
This pth file is too big to be include in this github repository.
So, Grab AI Challenge.ipynb will automatically download the pretrained model.
If you already had the pretrained pth file and wish to use it.
You can comment the code in fifth cell.

model_ft = models.resnet101(pretrained=True)

and uncomment the code.

model_ft = torch.load("resnet101-5d3b4d8f.pth")

Please make sure your pth file is in the same directory as the code.



## GPU

Grab AI Challenge.ipynb required GPU to run.
Please make sure you have a powerful GPU else it might take some time to train the model.

If you wish to run Grab AI Challenge.ipynb in CPU.
It is possible by edit the following lines.
In first cell, please edit device = torch.device("cuda:0") into device = torch.device("cpu")

In fourth cell, please remove all the .cpu().numpy()

I recommend this work to be run with powerful GPU.


###### Pre-processing.ipynb

This code pre-process the dataset and create the required folder and sub-folder.


###### Feature Engineering.ipynb

This code perform feature engineering.
This code will extract the bounding box from the image to reduce noise.


###### Grab AI Challenge.ipynb

This code will train resnet101 model and get the prediction for the test dataset.
The prediction include the best confidence result and also the confidence result for 196 classes.
This repository include the result csv file from pretrained resnet101 model that had been finetuned for 50 epochs.

###### best.csv

This csv file contained the best confidence prediction result together with the image filename for each prediction.

###### confidence.csv

This csv file contained the confidence prediction result from all the 196 classes together with the image filename for each prediction.


###### Get Dominant Colour of Car.ipynb

This code will get the dominant colour of the car for testing dataset.
The idea of this code is to get the dominant colour of the image.
Since the image after perform feature engineering will only contain mostly car.
So the dominant colour of the image after feature engineering will equal to the dominant colour of the car.
This code only tested with 30 images from test datasets due to the limitation of the computer resources.
So, the sample result csv files only have 30 results.

###### colour_test.csv

This csv file contained the colour name in words for the car.

###### rgb_colour.csv

This csv file contained the rgb colour code for the car.
