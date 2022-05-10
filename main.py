import numpy as np
import scipy.special
# import matplotlib.pyplot as plt

class neuralNetwork:
	# initialise the neural network
	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		# set number of nodes/ neurons in each layer
		self.inodes = input_nodes
		self.hnodes = hidden_nodes
		self.onodes = output_nodes

		# learning rate
		self.lr = learning_rate

		# initialize the weights 
		self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

		# sigmoid
		self.activation_function = lambda x: scipy.special.expit(x)
		pass

	# train train the neural network
	def train(self):
		pass

	# query the neural network
	def query(self, inputs_list):
		# convert inputs list to 2d array
		inputs = np.array(inputs_list, ndmin=2).T

		# hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		# output layer
		final_inputs = np.dot(self.who, inputs)
		final_outputs = self.activation_function(final_inputs)

		return final_outputs

# number of nodes
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

# learning rate = 0.3
learning_rate = 0.3

# calling NN
NN = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

print(NN.query([1.0, 0.5, -1.5]))
