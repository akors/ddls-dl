
# %%
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# %%
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# %%
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()


# %% convolutional layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# %% dense final layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# %%
model.summary()

# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy', 'sparse_categorical_crossentropy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))


# %%
fig, axes = plt.subplots(1, 2)

ax = axes.flat[0]

ax.plot(history.history['accuracy'], label='accuracy')
ax.plot(history.history['val_accuracy'], label = 'val_accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_ylim([0.5, 1])
ax.legend(loc='lower right')

ax = axes.flat[1]

ax.plot(history.history['sparse_categorical_crossentropy'], label='sparse_categorical_crossentropy')
ax.plot(history.history['val_sparse_categorical_crossentropy'], label = 'val_sparse_categorical_crossentropy')
ax.set_xlabel('Epoch')
ax.set_ylabel('CrossEntropy')
ax.legend(loc='lower right')

fig.set_size_inches(12, 8)

test_loss, test_acc, test_ce = model.evaluate(test_images,  test_labels, verbose=2)

print(f"Final test accuracy: {test_acc:.4f}")
print(f"Final test CrossEntropy: {test_ce:.4f}")
