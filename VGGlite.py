import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from keras import datasets, layers, models
import matplotlib.pyplot as plt
print(tf.__version__)
mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(16,(3,3), strides=(1,1),padding='same',activation='relu',input_shape=(28, 28, 1)))
model.add(layers.Conv2D(16,(3,3), strides=(1,1),padding='same',activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32,(3,3), strides=(1,1),padding='same',activation='relu'))
model.add(layers.Conv2D(32,(3,3), strides=(1,1),padding='same',activation='relu'))
model.add(layers.Conv2D(32,(3,3), strides=(1,1),padding='same',activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64,(3,3), strides=(1,1),padding='same',activation='relu'))
model.add(layers.Conv2D(64,(3,3), strides=(1,1),padding='same',activation='relu'))
model.add(layers.Conv2D(64,(3,3), strides=(1,1),padding='same',activation='relu'))
model.add(layers.MaxPooling2D((2, 2))) 
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(train_images.shape)
output = model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

plt.plot(output.history['accuracy'], label='accuracy')
plt.plot(output.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()



