# Load pickled data
import pickle
import numpy as np

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
validation_file = 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

# ------------------------------------------------------------------------------------------
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test,  y_test  = test['features'],  test['labels']


# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:3]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = np.max(y_train)

img_size = X_train.shape[1:3]


print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ------------------------------------------------------------------------------------------
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import random
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline

index = random.randint(0, len(X_train))
image_temp = X_train[index]

plt.figure(figsize=(1,1))
plt.imshow(image_temp)

print(y_train[index])



# ------------------------------------------------------------------------------------------
### Preprocess the data here. It is required to normalize the data.
### Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

X_train = (X_train - 128.)/128.
X_valid = (X_valid - 128.)/128.
X_test  =  (X_test - 128.)/128.



# ------------------------------------------------------------------------------------------
# Model Architecture
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128


# LeNet-5 ----------------------------------------------------------------------------------
from tensorflow.contrib.layers import flatten


# SAME Padding, the output height and width are computed as:
#
# out_height = ceil(float(in_height) / float(strides[1]))
#
# out_width = ceil(float(in_width) / float(strides[2]))
#
# VALID Padding, the output height and width are computed as:
#
# out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
#
# out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1


    # -----------------------------------------------------------------------------------------------
    # Layer 1: Convolutional. Input = 32x32x3 (RGB). Output = 16*16*32.
    # Same Padding --> out_height = in_height(32) / stride(2)
    conv1_W = tf.Variable(tf.truncated_normal(shape=(8, 8, 3, 32), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')



    # # -----------------------------------------------------------------------------------------------
    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)


    # -----------------------------------------------------------------------------------------------
    # Layer 3: Fully Connected. Input = 400. Output = n_classes.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, n_classes), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(n_classes))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)
    logits = fc1


    # # -----------------------------------------------------------------------------------------------
    # # Layer 4: Fully Connected. Input = 120. Output = 84.
    # fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    # fc2_b = tf.Variable(tf.zeros(84))
    # fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    #
    # # Activation.
    # fc2 = tf.nn.relu(fc2)
    #
    #
    # # -----------------------------------------------------------------------------------------------
    # # Layer 5: Fully Connected. Input = 84. Output = n_classes = 42
    # fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma))
    # fc3_b = tf.Variable(tf.zeros(n_classes))
    # logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


# def LeNet(x):
#     # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
#     mu = 0
#     sigma = 0.1
#
#     # Layer 1: Convolutional. Input = 32x32x3 (RGB). Output = 28x28x6.
#     conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma))
#     conv1_b = tf.Variable(tf.zeros(6))
#     conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
#
#     # Activation.
#     conv1 = tf.nn.relu(conv1)
#
#     # Pooling. Input = 28x28x6. Output = 14x14x6.
#     conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#
#     # Layer 2: Convolutional. Output = 10x10x16.
#     conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
#     conv2_b = tf.Variable(tf.zeros(16))
#     conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
#
#     # Activation.
#     conv2 = tf.nn.relu(conv2)
#
#     # Pooling. Input = 10x10x16. Output = 5x5x16.
#     conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#
#     # Flatten. Input = 5x5x16. Output = 400.
#     fc0 = flatten(conv2)
#
#     # Layer 3: Fully Connected. Input = 400. Output = 120.
#     fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
#     fc1_b = tf.Variable(tf.zeros(120))
#     fc1 = tf.matmul(fc0, fc1_W) + fc1_b
#
#     # Activation.
#     fc1 = tf.nn.relu(fc1)
#
#     # Layer 4: Fully Connected. Input = 120. Output = 84.
#     fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
#     fc2_b = tf.Variable(tf.zeros(84))
#     fc2 = tf.matmul(fc1, fc2_W) + fc2_b
#
#     # Activation.
#     fc2 = tf.nn.relu(fc2)
#
#     # Layer 5: Fully Connected. Input = 84. Output = n_classes = 42
#     fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma))
#     fc3_b = tf.Variable(tf.zeros(n_classes))
#     logits = tf.matmul(fc2, fc3_W) + fc3_b
#
#     return logits


# Features and Labels
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)



# Training Pipeline -------------------------------------------------------------------------------------
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# Model Evaluation -------------------------------------------------------------------------------------
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



# Train the Model -------------------------------------------------------------------------------------
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")



# Evaluate the Model -------------------------------------------------------------------------------------
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))












