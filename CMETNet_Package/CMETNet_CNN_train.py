
'''
 (c) Copyright 2022
 All rights reserved
 Programs written by Khalid A. Alobaid
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA
 Permission to use, copy, modify, and distribute this
 software and its documentation for any purpose and without
 fee is hereby granted, provided that this copyright
 notice appears in all copies. Programmer(s) makes no
 representations about the suitability of this
 software for any purpose.  It is provided "as is" without
 express or implied warranty.
'''


import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

try:
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('')
    
import numpy as np
import pandas as pd
import datetime
from skimage.transform import resize
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping
from CMETNet_CNN_model import CNN_Model


CME_FITS_list_filename = './data/ICMEs_C2_fits_list.csv'
CME_FITS_list = pd.read_csv(CME_FITS_list_filename)

work_path = './'

CME_images = []
CME_images_labels = []
Event_num_train = []
CME_images_test = []
CME_images_labels_test = []
Event_num_test = []

for row in range(0,len(CME_FITS_list)):  
    if (CME_FITS_list.Filepath[row][29:33]=='2013'):
        CME_TT=CME_FITS_list.transit_time[row]
        CME_FITS_path=work_path + CME_FITS_list.Filepath[row]
        image_file = get_pkg_data_filename(CME_FITS_path)
        hdu_list = fits.open(image_file)

        try:
            hdu_list[0].verify('fix+ignore')

        except:
            hdu_list[0].header[77]='BLANK'
            hdu_list[0].header[78]='BLANK'
            hdu_list[0].header[79]='BLANK'

            try:
                hdu_list[0].header[80]='BLANK'
            except:
                pass

            hdu_list[0].verify('fix+ignore')


        image_data = hdu_list[0].data
        CME_images.append(image_data)
        CME_images_labels.append(CME_TT)
        Event_num_train.append(int(CME_FITS_list.Filepath[row][25:28]))
        
for row in range(0,len(CME_FITS_list)):  
    if (CME_FITS_list.Filepath[row][29:33]=='2014'):
        CME_TT=CME_FITS_list.transit_time[row]
        CME_FITS_path=work_path + CME_FITS_list.Filepath[row]
        image_file = get_pkg_data_filename(CME_FITS_path)
        hdu_list = fits.open(image_file)

        try:
            hdu_list[0].verify('fix+ignore')

        except:
            hdu_list[0].header[77]='BLANK'
            hdu_list[0].header[78]='BLANK'
            hdu_list[0].header[79]='BLANK'

            try:
                hdu_list[0].header[80]='BLANK'
            except:
                pass

            hdu_list[0].verify('fix+ignore')


        image_data = hdu_list[0].data
        CME_images_test.append(image_data)
        CME_images_labels_test.append(CME_TT)
        Event_num_test.append(int(CME_FITS_list.Filepath[row][25:28]))
        
CME_images_rescaled = []
CME_images_rescaled_test = []

for image in CME_images:
    resized_image_data = resize(image,(256,256))
    pixels = np.asarray(resized_image_data)
    pixels = pixels.astype('float32')
    pixels /= pixels.max()
    resized_image_data_scalled = pixels.reshape(256,256,1)
    CME_images_rescaled.append(resized_image_data_scalled)
    
CME_images_rescaled = np.array(CME_images_rescaled)


for image in CME_images_test:
    resized_image_data = resize(image,(256,256))
    pixels = np.asarray(resized_image_data)
    pixels = pixels.astype('float32')
    pixels /= pixels.max()
    resized_image_data_scalled = pixels.reshape(256,256,1)
    CME_images_rescaled_test.append(resized_image_data_scalled)
    
CME_images_rescaled_test = np.array(CME_images_rescaled_test)

image_shape = (256, 256, 1)
model = CNN_Model(image_shape).cnn_model()
model.summary()
epoch_steps = 10


training_data = CME_images_rescaled[:]
training_data_labels = CME_images_labels[:]

X_train = training_data[:int(0.8 * len(training_data))]
y_train = np.array(training_data_labels[:int(0.8 * len(training_data_labels))])

X_val = training_data[int(0.8 * len(training_data)):]
y_val = np.array(training_data_labels[int(0.8 * len(training_data)):])

X_test = CME_images_rescaled_test
y_test = np.array(CME_images_labels_test)


model.compile(loss = 'mean_absolute_error', optimizer='adam')
history = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), verbose = 1, epochs=epoch_steps, batch_size=32)

pred = model.predict(X_test)
MAE_temp = round(mean_absolute_error(y_test, pred),2)



temp_result = pd.DataFrame(pred,columns=['results'])
temp_result['event_number']=Event_num_test
pred_results = temp_result.groupby('event_number')['results'].apply(list).reset_index(name='results')
temp_min = np.array(Event_num_test).min()
temp_max = np.array(Event_num_test).max()
np.arange(temp_min, temp_max+1, 1)

a = np.array(pred_results["event_number"])
b = np.arange(temp_min, temp_max+1, 1)
zero_events = np.setxor1d(a,b)
events_with_no_images = pd.DataFrame(zero_events,columns=['event_number'])
events_with_no_images['results']=0
final_CNN_results = pred_results.append(events_with_no_images).sort_values(by=['event_number'])

#safe to csv
temp_path = './results/'
filename1 = 'CMETNet_CNN_2014'
final_CNN_results.to_csv(temp_path+filename1, encoding='utf-8', index=False)

print("Done ------------------------------------ ")




