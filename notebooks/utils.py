import tf_keras_vis
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications.densenet import preprocess_input

def preprocess_image(img_path):
    img=cv2.imread(img_path)
    img=cv2.resize(img,(224,224))
    img=np.expand_dims(img,axis=0)
    processed_img=preprocess_input(img)
    return img,processed_img

def get_gradcam(model,img_path,class_idx):
    last_conv_layer='conv5_block15_1_relu'
    simg,sprocessed_img=preprocess_image(img_path)
    score=CategoricalScore([class_idx])
    gradcam=Gradcam(model,clone=False)
    cam=gradcam(score,sprocessed_img,penultimate_layer=last_conv_layer)
    return cam[0]

