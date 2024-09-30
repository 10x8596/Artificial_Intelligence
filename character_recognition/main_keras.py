import os
import tensorflow as tf 
from tensorflow.keras import regularizers
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 # to load and process images
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns

# load the data
mnist = tf.keras.datasets.mnist

# split into training and testing data
# where x is image data and y is classification
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixels data 
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
#
model = tf.keras.models.Sequential()
# Flatten the grid
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# Input layer
model.add(tf.keras.layers.Dense(128, activation='relu'))
# Hidden layers 
# Penalize large weights to prevent overfitting by L2 regularization
model.add(tf.keras.layers.Dense(128, activation='relu')) 
                                # kernel_regularizer=regularizers.l2(0.01)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
                                # kernel_regularizer=regularizers.l2(0.01)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# Output layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# Dropout layers to reduce overfitting by randomly turning off some neurons
# model.add(tf.keras.layers.Dropout(0.5))
# Normalize output of each layer for speed and accuracy
# model.add(tf.keras.layers.BatchNormalization())
# hyperparameter tune and compile the model
# opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Fit the model and train 
model.fit(x_train, y_train, epochs=8)

## Save the model 
# model.save('digit_recognition.keras')
## Load the model
# model = tf.keras.models.load_model('digit_recognition.keras')

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Confusion Matrix 
pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

def display_image(image, pred):
        plt.imshow(image[0], cmap=plt.cm.binary)
        plt.title(f"Predicted digit: {np.argmax(pred)}")
        plt.axis('off')
        plt.show()

fileNumber = random.randint(0, 99)
attempts = 0
while attempts < 100:
    try:
        filename = f"dataset/{fileNumber}.png"
        print(f"Showing file: {fileNumber}")
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # laod as grayscale
        if image is None:
            raise ValueError(f"Failed to load image: {filename}")

        # print(f"Original image shape: {image.shape}")
        # print(f"Pixel value range: {image.min()} to {image.max()}")

        # Resize image if not 28x28
        if image.shape != (28, 28):
            image = cv2.resize(image, (28, 28))
            print(f"Resized image shape: {image.shape}")

        # invert and normalize the image
        image = np.invert(np.array([image]))

        pred = model.predict(image)
        # return the neuron with the highest activation 
        print(f"The predicted digit is {np.argmax(pred)}")
        
        display_image(image, pred)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        fileNumber = random.randint(0, 99)
        attempts += 1 

print("End of program")
