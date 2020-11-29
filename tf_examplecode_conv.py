### Tensorflow example: a convolutional neural network MNIST classifier

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D
from matplotlib import image

# Load MNIST data and reshape from 3D to 4D
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape) # training images
print(y_train.shape) # their corresponding labels
print(x_test.shape) # test images
print(y_test.shape) # their corresponding labels

x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

print(x_train.shape) # training images reshaped to 1 dimension per image
print(x_test.shape) # test images reshaped to 1 dimension per image

# Convert pixel values between 0 and 255 to normalised values between 0 and 1
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# Define the model architecture
inputs = Input(shape=(28,28,1))
x = Conv2D(8, kernel_size=(3,3))(inputs)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(8, kernel_size=(3,3))(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Flatten()(x)
readout = Dense(10, activation=tf.nn.softmax)(x)
model = Model(inputs=inputs, outputs=readout)
model.summary()

# Define which loss function, optimisation function and metric(s) to use
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])

# Train and evaluate the model
model.fit(x=x_train,y=y_train, epochs=10)
model.evaluate(x_test, y_test)

# Single image inference
test_image = image.imread('test_img.png')
pred = model.predict(test_image.reshape(1, 28, 28, 1))
print('\nReadout node values (i.e. probabilities): {}'.format(pred))
print('\nPredicted digit for test image: {} \n'.format(pred.argmax()))

