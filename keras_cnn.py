""" Created in Apr 2019 by Paul A. Gureghian. """

""" This Python script has the code to setup a CNN in Keras. """

### Import Keras 
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

### Initialize the Sequential CNN model
classifier = Sequential() 

### Define the Convolution layer
classifier.add(Convolution2D(32, (3,3), input_shape=(64,64,3), activation='relu'))

### Define the Max Pooling layer
classifier.add(MaxPooling2D(pool_size= (2, 2)))

### Add a second Conv and MaxPool layer
classifier.add(Convolution2D(32, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size= (2, 2)))

### Add the Flatten layer
classifier.add(Flatten()) 

### Define the Dense layers
classifier.add(Dense(activation='relu', units=128))
classifier.add(Dense(activation='sigmoid', units=1))

### Compile the CNN 
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

### Fit the CNN to the training data
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/dataset/training_set',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('/dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         samples_per_epoch=8000,
                         nb_epoch=25,
                         validation_data=test_set,
                         nb_val_samples=2000) 











