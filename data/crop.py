from pycocotools.coco import COCO
import cv2
import os
import json
import numpy as np

def crop_imgs():
    coco_path  = '/home/Hacker_Davinci/Desktop/coco'
    crop_path = os.path.join(coco_path,'images/crop')
    if not os.path.exists(crop_path):
        os.makedirs(crop_path)
    coco = COCO(os.path.join(coco_path,'annotations/instances_train2017.json'))
    cat_objs = coco.loadCats(coco.getCatIds())

    cat_instances = {}

    for c in cat_objs:
        print(c['name'])
        cat_path = os.path.join(crop_path,f'{c["id"]}')
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)

        count = 0

        imgIds = coco.getImgIds(catIds = c['id'])
        cat_instances[c['id']] = []

        for i in imgIds:
            img_obj = coco.loadImgs(i)[0]
            img = cv2.imread(os.path.join(coco_path,'images/train2017',img_obj['file_name']),cv2.IMREAD_COLOR)
            annos = coco.loadAnns(coco.getAnnIds(i,c['id']))
            for a in annos:
                x,y,w,h = a['bbox']

                if w < 64 or h <64:
                    continue
                crop_img = img[int(y):int(y+h),int(x):int(x+w)]

                img_size = max(w,h)
                h_pad = int((img_size-h) // 2)
                w_pad = int((img_size-w) // 2)
                crop_img = np.pad(crop_img,((h_pad,h_pad),(w_pad,w_pad),(0,0)))
                crop_img = cv2.resize(crop_img,(64,64))

                cv2.imwrite(os.path.join(cat_path,f'{a["id"]}.png'),crop_img)

                cat_instances[c['id']].append(a["id"])

                count += 1
            
            if(count >=500):
                break

    with open(os.path.join(coco_path,'annotations/crop.json'),'w') as f:
        json.dump(cat_instances,f)
        

if __name__ == "__main__":
    crop_imgs()