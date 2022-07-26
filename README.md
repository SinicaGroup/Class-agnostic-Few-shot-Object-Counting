# Class agnostic Few shot Object Counting

This repository is the non-official pytorch implementation of a WACV 2021 Paper "Class-agnostic-Few-shot-Object-Counting". [Link](https://openaccess.thecvf.com/content/WACV2021/papers/Yang_Class-Agnostic_Few-Shot_Object_Counting_WACV_2021_paper.pdf)

In Proc. IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021
Shuo-Diao Yang, Hung-Ting Su, Winston H. Hsu, Wen-Chin Chen<sup>*</sup>

## Installation
Our code has been tested on Python 3.7.13 and PyTorch 1.8.1+cu101. Please follow the official instructions to setup your environment. See other required packages in `requirements.txt`.

## Data Preparation
We train and evaluate our methods on COCO dataset 2017. </br>
Please follow the instruction [here](https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9) to download the COCO dataset 2017 </br>
structure used in our code will be like : </br>
```
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
```
After download the data, please go to our code repository. </br>
Modify the variable in line 8  "coco_path" to your COCO dataset path.
```
cd CODE_DIRECTORY
python crop.py
```
After above instructions, structure used in our code will be like : </br>
```
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

```
## Acknowledgement
Thanks to the helpful discussion  (bmnet, cfocnet, oxford) </br>
The implementation  ... </br>

