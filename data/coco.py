import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from pycocotools.coco import COCO
import os
import cv2
import scipy.ndimage
import json

imgs_mean = [0.485, 0.456, 0.406]
imgs_std = [0.229, 0.224, 0.225]

cv2.setNumThreads(0) # disable multithread to avoid deadlocks

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class CountingkDataset(Dataset):

    def __init__(self, config,dataset_type, folds=[0,1,2]):
        super(CountingkDataset, self).__init__()
        self.config = config
        self.dataset_type = dataset_type
        self.coco_api = COCO(os.path.join(config.data.data_path,f'annotations/instances_{dataset_type}2017.json'))

        catIds = self.coco_api.getCatIds()
        self.select_catIds = []
        for s in folds:
            i = s
            while (i < len(catIds)):
                self.select_catIds.append(catIds[i])
                i += 4

        with open(os.path.join(self.config.data.data_path,'annotations/crop.json')) as f:
            self.cat_instances = json.load(f)

        self.select_img2cats_dict = {}
        self.select_imgIds = []

        for c in self.select_catIds:
            imgs = self.coco_api.getImgIds(catIds=c)
            for i in imgs:
                annIds = self.coco_api.getAnnIds(imgIds=i,
                                                catIds=c,
                                                iscrowd=False)
                anns = self.coco_api.loadAnns(annIds)
                if len(anns)>=5:
                    self.select_imgIds.append(i)
                    if i in self.select_img2cats_dict:
                        self.select_img2cats_dict[i].append(c)
                    else:
                        self.select_img2cats_dict[i] = [c]

        self.query_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256))
        ])
        self.reference_transform = transforms.ToTensor()

        self.length = len(self.select_imgIds)

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        select_imgId = self.select_imgIds[idx]
        img_obj = self.coco_api.loadImgs(select_imgId)[0]

        img = cv2.imread(os.path.join(self.config.data.data_path,f'images/{self.dataset_type}2017',img_obj['file_name']),cv2.IMREAD_COLOR)
        query_tensor = self.generate_query_tensor(img)

        select_catId = np.random.choice(self.select_img2cats_dict[select_imgId],1)
        select_catId = select_catId[0].item()
        references_tensor = self.generate_references_tensor(select_catId)

        target_densemap_tensor = self.get_target_densemap(img_obj,select_catId)

        return query_tensor, references_tensor, target_densemap_tensor

    def generate_query_tensor(self, img):
        img_size = img.shape
        max_img_size = max(img_size[0],img_size[1])
        ph = int((max_img_size - img_size[0])/2)
        pw = int((max_img_size - img_size[1])/2)

        pad_img = np.pad(img,((ph,ph),(pw,pw),(0,0)))
        pad_img = cv2.cvtColor(pad_img,cv2.COLOR_BGR2RGB)
        query_tensor = self.query_transforms(pad_img)

        return query_tensor

    def generate_references_tensor(self,select_catId,k=5):
        select_reference_imgIds = np.random.choice(self.cat_instances[str(select_catId)],k,True)

        references_tensor = []
        for ref_id in select_reference_imgIds:
            crop_img = cv2.imread(os.path.join(self.config.data.data_path,f'images/crop/{select_catId}/{ref_id}.png'),cv2.IMREAD_COLOR)

            crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)

            crop_tensor = self.reference_transform(crop_img)

            references_tensor.append(crop_tensor)
        reference_concat = torch.stack(references_tensor, 0)
        return reference_concat

    def get_target_densemap(self, img_obj, select_catId):


        max_img_size = max(img_obj['height'],img_obj['width'])
        scale_ratio = 256 / max_img_size
        ph = (max_img_size - img_obj['height']) /2
        ph = ph * scale_ratio
        pw = (max_img_size - img_obj['width']) /2
        pw = pw * scale_ratio

        density = np.zeros((256,256))

        anno_ids = self.coco_api.getAnnIds(int(img_obj['id']),select_catId,iscrowd=False)
        anno_objs = self.coco_api.loadAnns(anno_ids)

        ptx = []
        pty = []
        for anno_obj in anno_objs:
            bbox = np.array(anno_obj['bbox'])
            bbox = bbox * scale_ratio
            x = bbox[0] + pw
            y = bbox[1] + ph
            w = bbox[2]
            h = bbox[3]

            center_x = int(x+w/2)
            center_y = int(y+h/2)

            ptx.append(center_x)
            pty.append(center_y)


        density[pty,ptx] = 1
        density = scipy.ndimage.gaussian_filter(density, sigma=(5,5), mode='constant')
        density = torch.from_numpy(density.astype('float32'))

        return density.unsqueeze(0)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import argparse
    # import visdom
    # viz = visdom.Visdom()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    config = argparse.Namespace()
    config.data = argparse.Namespace()
    config.data.data_path = '/home/Hacker_Davinci/Desktop/coco'
    d = CountingkDataset(config)
    data_loader = DataLoader(d,1,shuffle=True)

    print("Total:", len(data_loader))
    it = iter(data_loader)
    for i in range(10):
        q,r,den = next(it)
        
        plt.imshow(den[0, 0].cpu().numpy())
        plt.savefig(f"/home/Hacker_Davinci/Desktop/Dataset/CFOCNet_3layers/CFOCNet_refector/test_dataloader/{i}_density.png")
        

        torchvision.utils.save_image(q, f"/home/Hacker_Davinci/Desktop/Dataset/CFOCNet_3layers/CFOCNet_refector/test_dataloader/{i}_query.png")
        # plt.imshow(q[0, 0].cpu().numpy())
        # plt.savefig(f"/home/Hacker_Davinci/Desktop/Dataset/CFOCNet_3layers/CFOCNet_refector/test_dataloader/{i}_query.png")
        
        # plt.imshow(r[0].cpu().numpy())
        # plt.savefig(f"/home/Hacker_Davinci/Desktop/Dataset/CFOCNet_3layers/CFOCNet_refector/test_dataloader/{i}_reference.png")
        torchvision.utils.save_image(r[0], f"/home/Hacker_Davinci/Desktop/Dataset/CFOCNet_3layers/CFOCNet_refector/test_dataloader/{i}_reference.png")
        
        # viz.heatmap(torch.flip(den.squeeze(),[0]))
        # viz.images(torch.clamp(q,0,1))
        # viz.images(torch.clamp(r.squeeze(),0,1))






