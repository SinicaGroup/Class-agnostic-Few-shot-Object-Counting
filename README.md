# Class agnostic Few shot Object Counting

This repository is the non-official pytorch implementation of a WACV 2021 Paper "Class-agnostic Few-shot-Object-Counting". [Link](https://openaccess.thecvf.com/content/WACV2021/papers/Yang_Class-Agnostic_Few-Shot_Object_Counting_WACV_2021_paper.pdf)

In Proc. IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021
Shuo-Diao Yang, Hung-Ting Su, Winston H. Hsu, Wen-Chin Chen<sup>*</sup>

![39_ref_good](https://user-images.githubusercontent.com/76461262/181033357-71dc9a34-7a78-4410-81d6-4dc7d74bedfd.png) </br>
<img src="https://user-images.githubusercontent.com/76461262/181033299-cda225d3-c964-4327-9d13-bdbdaa296af3.png" width="200" height="150" /> <img src="https://user-images.githubusercontent.com/76461262/181033407-cb571edc-cb2f-4f1a-9127-fb74cafc933c.png" width="300" height="225" /> <img src="https://user-images.githubusercontent.com/76461262/181033455-84efbbea-0656-4e47-b281-34e3eeb14482.png" width="300" height="225" /> </br>

![2_ref_good](https://user-images.githubusercontent.com/76461262/181036373-715da5ae-e150-4980-92e7-3434146e40e8.png) </br>
<img src="https://user-images.githubusercontent.com/76461262/181036450-ee30acc9-1521-4dd1-b5c3-562308dc7f8d.png" width="200" height="150" /> <img src="https://user-images.githubusercontent.com/76461262/181036669-6d0b78b4-8447-4f8c-9ac6-821351bc4f0b.png" width="300" height="225" /> <img src="https://user-images.githubusercontent.com/76461262/181036720-a7539696-bd5f-4886-aff6-4a343d390276.png" width="300" height="225" /> </br>

## Installation
Our code has been implemented on Python 3.8 and PyTorch 1.8.1+cu101. Please follow the instructions to setup your environment. See other required packages in `requirements.txt`.
````
conda create --name CFOCNet python=3.8
conda activate CFOCNet
pip install -r requirements.txt
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
````
If you have problem about installing **cocoapi**, come [here](https://github.com/philferriere/cocoapi) to find the official documentation.
## Getting Started
* [CFOCNet_demo.ipynb](CFOCNet_demo.ipynb) This notebook tests the detail implementations of CFOCNet, giving insights such as how the size of each tensor changes across each stage.
* [model](model) This directory contains all related modules of our CFOCNet implementation
* [Eval_Result](Eval_Result) This directory contains the ideal results during evaluation stage, where an example's predicted count and the density map aligns with the groundtruth.
## Data Preparation
We train and evaluate our methods on COCO dataset 2017. </br>
Please follow the instruction [here](https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9) to download the COCO dataset 2017 </br>
structure used in our code will be like : </br>
````
$PATH_TO_DATASET/
├──── images
│    ├──── train2017
│             |──── 118287 images (.jpg)
│
│    ├──── test2017
│             |──── 40670 images (.jpg)
│
│    ├──── val2017
│             |──── 5000 images (.jpg)
│
├──── annotations
│    ├──── captions_train2017.json
│
│    ├──── captions_val2017.json
│
│    ├──── instances_train2017.json
|
│    ├──── instances_val2017.json
│
│    ├──── person_keypoints_train2017.json
│
│    ├──── person_keypoints_val2017.json
````
After downloading the data, please navigate to our repository. </br>
Then, modify the variable "coco_path" in line 8  in [crop.py](data/crop.py) to your COCO dataset path.
````
cd CODE_DIRECTORY
python data/crop.py
````
After performing the above instructions, the structure of your coco dataset will be like : </br>
````
$PATH_TO_DATASET/
├──── images
│    ├──── train2017
│             |──── 118287 images (.jpg)
│
│    ├──── test2017
│             |──── 40670 images (.jpg)
│
│    ├──── val2017
│             |──── 5000 images (.jpg)
│
│    ├──── crop
│             |──── 80 directories which store all categories 500 images in coco dataset 2017 (for references images)
│
├──── annotations
│    ├──── captions_train2017.json
│
│    ├──── captions_val2017.json
│
│    ├──── crop.json
│
│    ├──── instances_train2017.json
|
│    ├──── instances_val2017.json
│
│    ├──── person_keypoints_train2017.json
│
│    ├──── person_keypoints_val2017.json

````

## Training
* Please go to [config.yaml](configs/config.yaml) to change the configs under "train". </br>
* To setup the training process, model configurations such as **epochs**, **batch_size**, and **result_path** can be tuned, all of which are stored in config.yaml</br>
* Modify run.sh to setup the training process ```python main.py --config=config.yaml --doc=doc_name --train```.
* doc_name can be any string you want.
* Execute the bash script through the below command:
````
cd CODE_DIRECTORY
bash run.sh
````
* After running the code, you will find your training logs under CODE_DIRECTORY/exp/logs/doc_name

## Testing
* Please go to [config.yaml](configs/config.yaml) to change the configs under "eval". </br>
* To setup the testing process, the configurations such as **checkpoint**, **sample**, and **image_folder** can be tuned, all of which are stored in config.yaml. </br>
* Modify run.sh to setup the testing process ```python main.py --config=config.yaml --doc=doc_name --test```
* doc_name can be any string you want.
* Execute the bash script through the below command
````
cd CODE_DIRECTORY
bash run.sh
````
* After running the code, you will find your testing logs under "CODE_DIRECTORY/exp/logs/doc_name".

## Implementation Details
* We simulate the project structure from [here](https://github.com/ermongroup/ncsnv2).
* We crop 500 reference images for each categories.
* For query image, instead of random crop, we resize it with aspect ratio and padding to 256 x 256.
* The default setting in our code is 5-shot learning, where each query image has 5 reference images to learn. 

## Acknowledgement
* Great thanks to the contributive discussions on the reproduction details with </br>
the author of [Class-Agnostic Few-Shot Object Counting](https://openaccess.thecvf.com/content/WACV2021/html/Yang_Class-Agnostic_Few-Shot_Object_Counting_WACV_2021_paper.html), Shuo-Diao Yang, </br>
the author of [Bilinear Matching Network](https://arxiv.org/abs/2203.08354), Min Shi, </br>
and the author of [Learning to Count Anything: Reference-less Class-agnostic Counting with Weak Supervision](https://arxiv.org/abs/2205.10203), Michael Hobley. </br> Their feedbacks have been really helpful, allowing us to replicate the model successfully. 
