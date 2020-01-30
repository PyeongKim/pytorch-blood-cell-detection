import numpy as np
import glob
from random import randint
import torch
import cv2
from torch.utils.data.dataset import Dataset
from pre_process import *


class CellDataset(Dataset):
    def __init__(self, image_path, annotation_path):
        # paths to all images and masks
        self.image_list = glob.glob(str(image_path) + str("/*"))
        self.data_len = len(self.image_list)
        self.annotation_path = annotation_path
        print('Train size:', self.data_len)

    def __getitem__(self, index):
        # Find image
        image_path = self.image_list[index]
        image_name = image_path[image_path.rfind('/')+1:]

        # Read image
        img_as_np = cv2.imread(image_path)
        img_as_np = cv2.cvtColor(img_as_np, cv2.COLOR_BGR2RGB) # H, W, C


        # --- Annotation --- #
        object_ = parse_annotation(self.annotation_path + '/' + image_name[:image_name.rfind("jpg")]+"xml")


        # --- crop --- #
        crop_ratio = randint(90, 100)/100
        if crop_ratio != 1:
          crop_loc, crop_size = define_crop_range(img_as_np, crop_ratio)
          img_as_np, object_ = crop_image(img_as_np, object_, crop_loc=crop_loc, crop_size=crop_size)


        # --- augmentation --- #

        # flip {0: vertical, 1: horizontal, 2: both, 3: none}
        flip_num = randint(0, 3)
        img_as_np = flip(img_as_np, flip_num)

        # Noise Determine {0: Gaussian_noise, 1: uniform_noise
        if randint(0, 1):
            # Gaussian_noise
            gaus_sd, gaus_mean = randint(0, 20), 0
            img_as_np = add_gaussian_noise(img_as_np, gaus_mean, gaus_sd)
        else:
            # uniform_noise
            l_bound, u_bound = randint(-20, 0), randint(0, 20)
            img_as_np = add_uniform_noise(img_as_np, l_bound, u_bound)

        # Brightness
        pix_add = randint(-20, 20)
        img_as_np = change_brightness(img_as_np, pix_add)

        """
        # Elastic distort {0: distort, 1:no distort}
        sigma = randint(6, 12)
        # sigma = 4, alpha = 34
        img_as_np = add_elastic_transform(img_as_np, alpha=34, sigma=sigma, pad_size=20)
        """

        # --- Resize --- #
        img_as_np, object_ = resize_image(img_as_np, object_, 448)
      

        # Normalize image
        img_as_np = img_as_np/255

        # Convert numpy array to tensor
        img_as_np = img_as_np.transpose(2, 0, 1)
        img_as_tensor = torch.from_numpy(img_as_np).float()
        object_as_tensor = torch.tensor(object_)

        print(img_as_np.shape)
 
        return img_as_tensor, object_as_tensor

    def __len__(self):
        return self.data_len
    
    def collate_fn(self, batch):
      images = list()
      objects_ = list()
      for b in batch:
          images.append(b[0])
          objects_.append(b[1])
      images = torch.stack(images, dim=0)
      

      return images, objects_
      
      
if __name__ == "__main__":
  image_path = "/content/gdrive/My Drive/Colab Notebooks/data/JPEGImages"
  annotation_path = "/content/gdrive/My Drive/Colab Notebooks/data/Annotations"
  data = CellDataset(image_path, annotation_path)
  data.__getitem__(10)
