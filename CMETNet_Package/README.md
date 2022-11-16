## Predicting CME arrival time through data integration and ensemble learning<br>
This README explains the requirements and getting started to run the CME transit time prediction using the Deep Learning network CMETNet.

## Authors
Khalid A. Alobaid, Yasser Abduallah, Jason T. L. Wang, and Haimin Wang

## Abstract

The Sun constantly releases radiation and plasma into the heliosphere. 
Sporadically, the Sun launches solar eruptions such as flares and coronal mass ejections (CMEs). 
CMEs carry away a huge amount of mass and magnetic flux with them. An Earth-directed CME can cause serious consequences to the human system. 
It can destroy power grids/pipelines, satellites, and communications. 
Therefore, accurately monitoring and predicting CMEs is important to minimize damages to the human system. 
In this study we propose an ensemble learning approach, named CMETNet, 
for predicting the arrival time of CMEs from the Sun to the Earth. 
We collect and integrate eruptive events from two solar cycles, #23 and #24, from 1996 to 2021 with a total of 363 geoeffective CMEs. 
The data used for making predictions include CME features, 
solar wind parameters and CME images obtained from the SOHO/LASCO C2 coronagraph. 
Our ensemble learning framework comprises regression algorithms for numerical data analysis and a convolutional neural network for image processing. 
Experimental results show that CMETNet performs better than existing machine learning methods reported in the literature, 
with a Pearson product-moment correlation coefficient of 0.83 and a mean absolute error of 9.75 h.

For the latest updates of the tool refer to https://github.com/deepsuncode/CMETNet

## Prerequisites
| Library | Version | Description  |
|---|---|---|
| python| 3.8.13 | Programming language|
| cuda| 10.1 | Toolkit to develop, optimize, and deploy your applications on GPU|
| cudnn| 7.6.5 | Deep Neural Network library|
| numpy| 1.19.1 | Mathematical functions to operate on arrays|
| scikit-learn| 1.0.2 | Machine learning algorithms|
| scikit-image| 0.19.2 | Image preprocessing |
| pandas|1.2.4 | Data loading and manipulation|
| tensorflow| 2.4.1 | Machine learning ecosystem|
| tensorflow-gpu| 2.4.1 | GPU utlization for Deep learning algorithms |
| xgboost| 1.5.0 | Gradient boosted trees algorithm |
| astropy| 4.0.2 | Framework for commonly-used astronomy tools |
| matplotlib| 3.6.2 | Visutalization tool |

In order to use the code to run some predictions, you should use the exact version of Python and its packages stated above. 
Other versions are not tested, but they should work if you have the environment set properly to run deep learning jobs.


## Installation on local machine
To install the required packages, you may use Python package manager "pip" as follow:
1.	Copy the above packages into a text file,  ie "requirements.txt"
2.	Execute the command: 

Type:

	pip install -r requirements.txt

Note: There is a requirements file already created for you to use that includes all packages with their versions. The files are located in the root directory of the CMETNet.
Note: Python packages and libraries are sensitive to versions. Please make sure you are using the correct packages and libraries versions as specified above.

Cuda Installation Package:
You may download and install Cuda v 10.1 from https://developer.nvidia.com/cuda-10.1-download-archive-base

## Package Structure
After downloading the zip files from github repository: https://github.com/deepsuncode/CMETNet the CMETNet package includes the following folders and files:

| File | Description  |
|---|---|
| README.md | this  file | 
|  requirements.txt  | includes Python required packages for Python version 3.8.13| 
|  data   | includes data that can be used for training and prediction| 
|  results | will include the prediction result file(s)| 
|  CMETNet_CNN_train.py | Python program to train/predict the CMETNet CNN model| 
|  CMETNet_COMB_train.py  | Python program to train/predict CMETNet COMB models| 
|  results_ensemble.py | Python program to ensemble the 5 models and produce CMETNet results| 
|  pretrained_models_results.py | Python program to print pretrained results| 
 
 
 
## Running a Train/Prediction Task:
To run a train task, you should type: 

	python CMETNet_CNN_train.py
	
and

	python CMETNet_COMB_train.py
	
Without any option will this will train all the models using data samples.

## Running a Testing Task:
To show the results of CMETNet, you should type:

	python results_ensemble.py
	
This will ensemble the results made by the models and show the results and the PCCMM/MAE metrics, save the CMETNet results in the results folder.

