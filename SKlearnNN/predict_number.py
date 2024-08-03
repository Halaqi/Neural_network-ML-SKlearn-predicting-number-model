import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load data
train_x = np.loadtxt("train_X.csv", delimiter=',').T
train_y = np.loadtxt("train_label.csv", delimiter=',').T
test_x = np.loadtxt("test_X.csv", delimiter=',').T
test_y = np.loadtxt("test_label.csv", delimiter= ',').T

# Check shapes
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# Convert labels to integer format
train_y = np.argmax(train_y, axis=0)
test_y = np.argmax(test_y, axis=0)

# Define and train the model
mlp = MLPClassifier(hidden_layer_sizes=(1000,), activation='tanh', solver='adam', learning_rate_init=0.02, max_iter=500)

mlp.fit(train_x.T, train_y)

# Evaluate the model
train_pred = mlp.predict(train_x.T)
test_pred = mlp.predict(test_x.T)

train_accuracy = accuracy_score(train_y, train_pred) * 100
test_accuracy = accuracy_score(test_y, test_pred) * 100

print('The accuracy of our training data is', train_accuracy, "%")
print('The accuracy of our testing data is', test_accuracy, "%")

# Display image and predict the output
idx = int(random.randrange(0, test_x.shape[1]))
plt.imshow(test_x[:, idx].reshape((28, 28)), cmap='gray')
plt.show()

predicted_label = mlp.predict(test_x[:, idx].reshape(1, -1))

print("Our model says it is:", predicted_label[0])
