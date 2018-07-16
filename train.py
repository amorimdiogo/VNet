"""
Diogo Amorim, 2018-07-09
V-Net training
"""

from keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from utils import *
import numpy as np
from vnet import vnet


save_dir = "/home/guest/PycharmProjects/tese/Vnet/dataset/"
train_dir = save_dir + "train_data.h5"
val_dir = save_dir + "val_data.h5"
weights_dir = save_dir + "weights_vnet.h5"

x_train, y_train = load_dataset(train_dir)
x_val, y_val = load_dataset(val_dir)

callbacks = list()
callbacks.append(ModelCheckpoint(weights_dir, monitor='val_loss', save_weights_only=True, save_best_only=True))
callbacks.append(CSVLogger(save_dir + "training.log", append=True))
callbacks.append(ReduceLROnPlateau(factor=0.5, patience=2, verbose=1))
callbacks.append(EarlyStopping(patience=2))

model = vnet(input_size=(256, 256, 64, 1))

model.fit(x_train, y_train,
          batch_size=1, epochs=8,
          validation_data=(x_val, y_val), callbacks=callbacks)

# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2)
#
# datagen.fit(x_train[:,:,:,:,0])
#
# model.fit(datagen.flow(x_train, y_train, batch_size=4, seed=1),
#           steps_per_epoch=len(x_train) / 4, epochs=8, validation_data=(x_val, y_val), callbacks=callbacks
#           )

# model.save_weights(weights_dir)
