import os
import torch
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2
import xml.etree.ElementTree as ETree
import torchvision.transforms.functional as TF

cell_subtypes = ("RBC", "WBC", "Platelets")
subtypes_map = {key: i+1 for i, key in enumerate(cell_subtypes)}
distinct_colors = ['#3cb44b', '#ffe119', '#0082c8']
subtypes_color_map = {key: distinct_colors[i] for i, key in enumerate(subtypes_map)}

def parse_annotation(xml_path):
  """
  Args: 
    xml_path (str): path to xml file
  Return:
    obejct_ (list): location of the ground truth bounding boxes and corresponding labels 
  """

  tree = ETree.parse(xml_path)
  root = tree.getroot()

  object_ = list()

  for cell in root.iter("object"):
    # get the tree with the heading "name"
    subtype = cell.find("name").text.strip()
    assert subtype in subtypes_map, "undefined label detected"

    box = cell.find("bndbox")
    xmin = int(box.find("xmin").text)
    xmax = int(box.find("xmax").text)
    ymin = int(box.find("ymin").text)
    ymax = int(box.find("ymax").text)
    label = subtypes_map[subtype]
    object_.append([xmin, ymin, xmax, ymax, label])
    
  return object_

def normalization(image, mean, std):
  """
  Args:
    image (numpy) : input image (H,W,C)
    mean (list) : mean of each channel of the image (R,G,B)
    std (list) : std of each channel of the image (R,G,B)
  Return:
    normalized_image (numpy) : normalized image (H,W,C)
  """
  normalized_image = np.zeros(image.shape)
  for c in range(image.shape[-1]):
    normalized_image[:,:,c] = (image[:,:,c] - mean[c])/std[c]
  return normalized_image

def resize_image(image, object_, target_size=448):
   """
  Args:
    image (numpy array) : input image (H, W, C)
    object_ (list) : location of the ground truth bounding boxes and corresponding labels
    target_size (int) : target size to resize, we use 448x448 as default
  Return:
    resized_image (numpy array) : resized image (H, W, C)
    object_ (list) : location of the ground truth bounding boxes that are rescaled 
  """
  height, width = image.shape[:2] 
  resized_image = cv2.resize(image,(target_size, target_size))
  width_ratio, height_ratio = width/target_size, height/target_size
  for i in range(len(object_)):
    object_[i][0] = round(object_[i][0]*width_ratio) # xmin
    object_[i][2] = round(object_[i][2]*width_ratio) # xmax
    object_[i][1] = round(object_[i][1]*height_ratio) # ymin
    object_[i][3] = round(object_[i][3]*height_ratio) # ymax

  return resized_image, object_


def define_crop_range(image, crop_ratio):
  """
  Args:
    image (numpy array): original image (H,W,C)
    crop_ratio (float): range from 0.9 to 1
  Return:
    crop_location: x and y
  """
  height, width = image.shape[:2]
  crop_height, crop_width = round(height*crop_ratio), round(width*crop_ratio)
  return (width - crop_width, height - crop_height), (crop_width, crop_height)
  

def crop_image(image, object_, crop_size=(40, 80), crop_loc=(0, 0)):
  """
  Crop the image:
    Args:
      image (numpy array) : numpy array (H,W,C)
      object (list): location of the bounding box and label (xmin, ymin, xmax, ymax, label)
      crop_size (tuple or list) : the desired crop size (W, H)
      crop_loc (tuple or list) : the desired location to crop (x, y)
    return:
      image_cropped (numpy array) : cropped image of specific demension (H,W,C)
      object_ (list) : location of the ground truth bounding boxes cropped 
  """
  # crop the image
  xmin, xmax = crop_loc[0], crop_loc[0]+crop_size[0]
  ymin, ymax = crop_loc[1], crop_loc[1]+crop_size[1]
  image_cropped =  image[ymin:ymax, xmin:xmax]
  for i in range(len(object_)):
    # crop the bounding box
    object_[i][0] = max(object_[i][0], xmin)
    object_[i][1] = max(object_[i][1], ymin)
    object_[i][2] = min(object_[i][2], xmax)
    object_[i][3] = min(object_[i][3], ymax)

  return image_cropped, object_


def flip(image, option_value):
    """
    Args:
        image : numpy array of image
        option_value = random integer between 0 to 3
    Return :
        image : numpy array of flipped image
    """
    if option_value == 0:
        # vertical
        image = np.flip(image, option_value)
    elif option_value == 1:
        # horizontal
        image = np.flip(image, option_value)
    elif option_value == 2:
        # horizontally and vertically flip
        image = np.flip(image, 0)
        image = np.flip(image, 1)
    else:
        image = image
        # no effect
    return image

def add_elastic_transform(image, alpha, sigma):
    """
    Args:
        image : numpy array of image
        alpha : α is a scaling factor
        sigma :  σ is an elasticity coefficient
        random_state = random integer
        Return :
         : elastically transformed numpy array of image
    """
    assert image.shape[-1] == 3, "the elastic transform does not support gray scale image"
    random_state = np.random.RandomState(None)

    pad_size = 20
    image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT_101)

    shape = image.shape[:2]
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices =  np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    transformed_img = np.empty_like(image)
    for i in range(image.shape[-1]):
        transformed_img[:, :, i] = map_coordinates(image[:, :, i], indices).reshape(shape)
    return transformed_img[pad_size:-pad_size, pad_size:-pad_size,:]
    


def add_gaussian_noise(image, mean=0, std=1):
    """
    Args:
        image : numpy array of image
        mean : pixel mean of image
        standard deviation : pixel standard deviation of image
    Return :
        image : numpy array of image with gaussian noise added
    """
    gaus_noise = np.random.normal(mean, std, image.shape)
    image = image.astype("int16")
    noise_img = image + gaus_noise
    image = ceil_floor_image(image)
    return noise_img


def add_uniform_noise(image, low=-10, high=10):
    """
    Args:
        image : numpy array of image
        low : lower boundary of output interval
        high : upper boundary of output interval
    Return :
        image : numpy array of image with uniform noise added
    """
    uni_noise = np.random.uniform(low, high, image.shape)
    image = image.astype("int16")
    noise_img = image + uni_noise
    image = ceil_floor_image(image)
    return noise_img


def change_brightness(image, value):
    """
    Args:
        image : numpy array of image
        value : brightness
    Return :
        image : numpy array of image with brightness added
    """
    image = image.astype("int16")
    image = image + value
    image = ceil_floor_image(image)
    return image


def ceil_floor_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 with ceilling(maximum 255) and flooring(minimum 0)
    """
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image

  
