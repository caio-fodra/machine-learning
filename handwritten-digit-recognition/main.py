# importing the hand written digit dataset
from sklearn import datasets
from sklearn.metrics import accuracy_score

# digit contain the dataset
digits = datasets.load_digits()

# dir function use to display the attributes of the dataset
dir(digits)

# outputting the picture value as a series of numbers
print(digits.images[0])

# importing the matplotlib libraries pyplot function
import matplotlib.pyplot as plt
# defining the function plot_multi

def plot_multi(i):
	nplots = 16
	fig = plt.figure(figsize=(15, 15))
	for j in range(nplots):
		plt.subplot(4, 4, j+1)
		plt.imshow(digits.images[i+j], cmap='binary')
		plt.title(digits.target[i+j])
		plt.axis('off')
	# printing the each digits in the dataset.
	plt.show()

	plot_multi(0)
	
# converting the 2 dimensional array to one dimensional array
y = digits.target
x = digits.images.reshape((len(digits.images), -1))

# gives the  shape of the data
x.shape

# printing the one-dimensional array's values
x[0]

# Very first 1000 photographs and
# labels will be used in training.
x_train = x[:1000]
y_train = y[:1000]

# The leftover dataset will be utilised to
# test the network's performance later on.
x_test = x[1000:]
y_test = y[1000:]

# importing the MLP classifier from sklearn
from sklearn.neural_network import MLPClassifier

# calling the MLP classifier with specific parameters
mlp = MLPClassifier(hidden_layer_sizes=(15,),
                    activation='logistic',
                    alpha=1e-4, solver='sgd',
                    tol=1e-4, random_state=1,
                    learning_rate_init=.1,
                    verbose=True)

mlp.fit(x_train, y_train)

fig, axes = plt.subplots(1, 1)
axes.plot(mlp.loss_curve_, 'o-')
axes.set_xlabel("number of iteration")
axes.set_ylabel("loss")
plt.show()

predictions = mlp.predict(x_test)
predictions[:50]
y_test[:50]

# calculating the accuracy with y_test and predictions
accuracy_score(y_test, predictions)
