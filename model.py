import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

inputs = Input(shape=(128, 128, 3))

conv1 = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(inputs)
conv2 = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(conv1)
pool1 = MaxPooling2D(pool_size=2)(conv2)

conv3 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(pool1)
conv4 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu')(conv3)
pool2 = MaxPooling2D(pool_size=2)(conv4)

conv5 = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(pool2)
conv6 = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(conv5)
conv7 = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(conv6)
pool3 = MaxPooling2D(pool_size=2)(conv7)

conv8 = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(pool3)
conv9 = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(conv8)
conv10 = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(conv9)
pool4 = MaxPooling2D(pool_size=2)(conv10)

conv11 = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(pool4)
conv12 = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(conv11)
conv13 = Conv2D(512, (3, 3), strides=1, padding='same', activation='relu')(conv12)
pool5 = MaxPooling2D(pool_size=2)(conv13)

flat = Flatten()(pool5)

fc1 = Dense(1024, activation='relu')(flat)
fc2 = Dense(512, activation='relu')(fc1)

outputs = Dense(2, activation='softmax')(fc2)

VGG16_model = Model(inputs=inputs, outputs=outputs)

VGG16_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.1),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

VGG16_model.summary()