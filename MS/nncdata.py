import tensorflow as tf
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import np_utils

class NncData:

    def get_csv_list(self,fname):
        csv_file = open(fname, "r", encoding="utf8", errors="", newline="" )
        dset = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
        next(dset)
        return dset

    def load_nnc_image(self,fname,gray=False,width=150,height=150,category=2,test=0.2, seed=14365):
        x_data = []
        y_label = []
        tx = []
        ty = []
        dcsv = self.get_csv_list(fname)
        for r in dcsv:
            if(gray == True):
                temp_img = load_img(r[0], color_mode = "grayscale",target_size=(height,width))
            else:
                temp_img = load_img(r[0],target_size=(height,width))
            img_array  = img_to_array(temp_img)
            
            tx.append(img_array)
            ty.append(r[1])

        x_data = np.asarray(tx)
        y_label = np.asarray(ty)

        x_data = x_data.astype('float32')
        x_data = x_data / 255.0

        y_label = np_utils.to_categorical(y_label, category)

        return train_test_split(x_data, y_label, test_size=test, random_state=seed)