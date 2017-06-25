from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

class Model :
    def CreateModel(self) :
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(32), Activation="relu")
        #self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])



    def TrainModel(self) :
        batch_size = 16
        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1./255)

        # this is a generator that will read pictures found in
        # subfolers of 'data/train', and indefinitely generate
        # batches of augmented image data
        train_generator = train_datagen.flow_from_directory(
                'data/train',  # this is the target directory
                target_size=(48, 48),  # all images will be resized to 48x48
                batch_size=batch_size,
                class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

        # this is a similar generator, for validation data
        validation_generator = test_datagen.flow_from_directory(
                'data/validation',
                target_size=(48, 48),
                batch_size=batch_size,
                class_mode='binary')

        self.model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
        self.model.save_weights('first_try.h5')  # always save your weights after training or during training
