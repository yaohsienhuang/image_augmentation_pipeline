import cv2
import numpy as np
import matplotlib.pyplot as plt
from augmentations.IaaAugmentation import IaaAugmentation
from augmentations.CustomAugmentation import RandomBrightness, RandomContrast

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image= transform(image)
        return image

class TrainAugmentations:
    '''
    img_custom : Functions to strengthen or weaken the contrast/brightness in each image.
    img_iaa : Functions to augmentations with a given probility in the Sequence of iaa.
    '''
    def __init__(self):
        self.aug_pipeline = Compose([
            RandomBrightness(20),
            RandomContrast(0.5, 1.2),
            IaaAugmentation(0.8),
        ])
        
    def __call__(self, image):
        image= self.aug_pipeline(image)
        return image
    
    
if __name__=='__main__':
    img = cv2.imread('sample/sample.jpg')  
    img = cv2.resize(img, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img= np.array(img, dtype=np.float32)
    img_origin=img.copy()
    
    Augmenation=TrainAugmentations()
    
    plt.figure(figsize=(20,20))
    for i in range(10):
        output=Augmenation(img)
        plt.subplot(1, 11, 1)
        plt.imshow(img_origin.astype(np.uint8))
        plt.axis('off')
        plt.subplot(1, 11, i+2)
        plt.imshow(output.astype(np.uint8))
        plt.axis('off')
    
    plt.show()
    
    