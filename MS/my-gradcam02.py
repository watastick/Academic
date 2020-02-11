# coding:utf-8

import pandas as pd
import numpy as np
import cv2
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img, save_img
from keras.models import load_model

K.set_learning_phase(1) #set learning phase



def Grad_Cam(input_model, x, layer_name):
    '''
    Args:
       input_model: model objects
       x: image(array)
       layer_name: convolutional layer name

    Returns:
       jetcam: heatmap images.(array)

    '''

    # Preprocessing
    X = np.expand_dims(x, axis=0)

    X = X.astype('float32')
    preprocessed_input = X / 255.0


    # calculate prediction classes

    predictions = model.predict(preprocessed_input)
    class_idx = np.argmax(predictions[0])
    class_output = model.output[:, class_idx]


    # get gradient

    conv_output = model.get_layer(layer_name).output
    grads = K.gradients(class_output, conv_output)[0]
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0], grads_val[0]

    # multiply mean weight by layer output.
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)


    # heatmap

    cam = cv2.resize(cam, (150, 150), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)
    jetcam = (np.float32(jetcam) + x / 2)  

    return jetcam


image_name = '*****.png' 

model = load_model("*****.h5")
x = img_to_array(load_img(image_name, target_size=(150, 150)))
array_to_img(x)


image = Grad_Cam(model, x, '*****')
array_to_img(image)

save_img('grad-cam_%s' %image_name, image)


'''
cv2.imshow('result_image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''