import albumentations as A
import numpy as np
import cv2

class DarkAug(object):
    """
    Extreme dark augmentation aiming at Aachen Day-Night
    """

    def __init__(self):
        self.augmentor = A.Compose([
            A.RandomBrightnessContrast(p=0.75, brightness_limit=(-0.6, 0.0), contrast_limit=(-0.5, 0.3)),
            A.Blur(p=0.1, blur_limit=(3, 9)),
            A.MotionBlur(p=0.2, blur_limit=(3, 25)),
            A.RandomGamma(p=0.1, gamma_limit=(15, 65)),
            A.HueSaturationValue(p=0.1, val_shift_limit=(-100, -40))
        ], p=0.75)

    def __call__(self, x):
        return self.augmentor(image=x)['image']


class MobileAug(object):
    """
    Random augmentations aiming at images of mobile/handhold devices.
    """

    def __init__(self):
        self.augmentor = A.Compose([
            A.MotionBlur(p=0.25),
            A.ColorJitter(p=0.5),
            A.RandomRain(p=0.1),  # random occlusion
            A.RandomSunFlare(p=0.1),
            A.JpegCompression(p=0.25),
            A.ISONoise(p=0.25)
        ], p=1.0)

    def __call__(self, x):
        return self.augmentor(image=x)['image']
    
class RGBThermalAug(object):
    """
    Pseudo-thermal image augmentation 
    """

    def __init__(self):
        self.blur =  A.Blur(p=0.7, blur_limit=(2, 4))
        self.hsv = A.HueSaturationValue(p=0.9, val_shift_limit=(-30, +30), hue_shift_limit=(-90,+90), sat_shift_limit=(-30,+30))

        # Switch images to apply augmentation
        self.random_switch = True

        # parameters for the cosine transform
        self.w_0 = np.pi * 2 / 3
        self.w_r = np.pi / 2
        self.theta_r = np.pi / 2

    def augment_pseudo_thermal(self, image):
        
        # HSV augmentation
        image = self.hsv(image=image)["image"]
        
        # Random blur
        image = self.blur(image=image)["image"]
        
        # Convert the image to the gray scale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  

        # Normalize the image between (-0.5, 0.5)
        image = image / 255 - 0.5 # 8 bit color

        # Random phase and freq for the cosine transform
        phase = np.pi / 2 + np.random.randn(1) * self.theta_r
        w = self.w_0 + np.abs(np.random.randn(1)) *  self.w_r
        
        # Cosine transform
        image = np.cos(image * w + phase) 

        # Min-max normalization for the transformed image
        image = (image - image.min()) / (image.max() - image.min()) * 255

        # 3 channel gray
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        return image

    def __call__(self, x, image_num):
        if image_num==0:
            # augmentation for RGB image can be added here
            return  x 
        elif image_num==1:
            # pseudo-thermal augmentation
            return self.augment_pseudo_thermal(x)
        else:
            raise ValueError(f'Invalid image number: {image_num}')


def build_augmentor(method=None, **kwargs):

    if method == 'dark':
        return DarkAug()
    elif method == 'mobile':
        return MobileAug()
    elif method == "rgb_thermal":
        return RGBThermalAug()
    elif method is None:
        return None
    else:
        raise ValueError(f'Invalid augmentation method: {method}')


if __name__ == '__main__':
    augmentor = build_augmentor('FDA')