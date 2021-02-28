from albumentations import CLAHE, Blur
import cv2
import matplotlib.pyplot as plt
import PIL
import numpy as np
from IPython.display import Image

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)

#Read file
file='zdj1.png'
img = PIL.Image.open(file)
#Enhance colors
converter = PIL.ImageEnhance.Color(img)
img = converter.enhance(3)
img = np.array(img2)
#CLAHE
aug = CLAHE(p=0.7)
img=aug(image=img2)['image']
#NLM
img = cv2.fastNlMeansDenoisingColored(img2,None,10,10,7,21)
visualize(img)
