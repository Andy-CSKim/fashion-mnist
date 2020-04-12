

from utils import mnist_reader
import numpy as np

print("=== reading dataset ===")

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

print(X_train.shape, X_train.ndim, X_train.size)
print(y_train.shape, y_train.ndim, y_train.size)

print(len(X_train))

#print(X_train[0]) # [0] .. [59999], [x] = 784 pts
#print(y_train[:20]) # [0] .. [59999], [x] = label
#print(y_test[:20]) # [0] .. [59999], [x] = label

#print(X_train[0][:100])

#print(type(y_train[0]))
#print(type(X_train[0][0]))
# import matplotlib.pyplot as plt

class myNeuralNet:

    def __init__(self, num_inodes, num_hnodes, num_onodes, learning_rate):
        self.inodes = num_inodes
        self.hnodes = num_hnodes
        self.onodes = num_onodes
        self.lrate = learning_rate

        # define weight
        self.W_ih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.W_ho = np.random.rand(self.onodes, self.hnodes) - 0.5
        print("W_ih, shape = ", self.W_ih.shape)
        print("W_ho, shape = ", self.W_ho.shape)

        self.temp_ih = np.zeros((self.hnodes, self.inodes))
        self.temp_ho = np.zeros((self.onodes, self.hnodes))

    def _activation(self, x):
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
        tanh = lambda x: 2.0 * sigmoid(2.0 * x) - 1.0

        return sigmoid(x)

    def _dev_activation(self, x):
        dev_sigmoid = lambda x: x * (1.0 - x)
        dev_tanh = lambda x: (1.0 - x) * (1.0 + x)

        return dev_sigmoid(x)

    def _forward(self, weights, node_in):
        return self._activation(np.dot(weights, node_in))

    def _delta_weight(self, node_in, node_out, error):
        return np.dot(error * self._dev_activation(node_out), node_in.T)

    def fit(self, in_list, target_list):
        inputs = np.array(in_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        hiddens_o = self._forward(self.W_ih, inputs)
        outputs_o = self._forward(self.W_ho, hiddens_o)

        # error backpropagation and update weights
        error_o = targets - outputs_o
        self.W_ho += self.lrate * self._delta_weight(hiddens_o, outputs_o, error_o)
        error_h = np.dot(self.W_ho.T, error_o)
        self.W_ih += self.lrate * self._delta_weight(inputs, hiddens_o, error_h)

    def fit_by_GD(self, in_list, target_list):
        inputs = np.array(in_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        hiddens_o = self._forward(self.W_ih, inputs)
        outputs_o = self._forward(self.W_ho, hiddens_o)

        # error backpropagation and update weights
        error_o = targets - outputs_o
        self.temp_ho += self._delta_weight(hiddens_o, outputs_o, error_o)
        error_h = np.dot(self.W_ho.T, error_o)
        self.temp_ih += self._delta_weight(inputs, hiddens_o, error_h)

    def updateWeight(self, length):
        # need to rescaling self.temp
        # print("temp_ho (min, max) = ", np.min(self.temp_ho), np.max(self.temp_ho))
        # print("temp_ih (min, max) = ", np.min(self.temp_ih), np.max(self.temp_ih))
        def _rescalingWeight(weight):
            # normalization
            weight -= np.min(weight)
            weight /= np.max(weight)
            weight *= 0.98
            weight -= 0.5

            # _rescalingWeight(self.temp_ho)

        # _rescalingWeight(self.temp_ih)

        self.temp_ho /= length
        self.temp_ih /= length

        self.W_ho += self.lrate * self.temp_ho
        self.W_ih += self.lrate * self.temp_ih

        self.temp_ih *= 0.
        self.temp_ho *= 0.

    def predict(self, in_list):
        # in_list is simply python list
        inputs = np.array(in_list, ndmin=2).T
        # print(inputs); print()

        hiddens_o = self._forward(self.W_ih, inputs)
        outputs_o = self._forward(self.W_ho, hiddens_o)

        return outputs_o


# init

print(" === instantiation of NN === ")
inodes = 784
hnodes = 100
onodes = 10
learning_rate = 0.1

np.random.seed(100)
myNN = myNeuralNet(inodes, hnodes, onodes, learning_rate)
print(myNN.W_ih[:5])


# training
print(" === training === ")

epoch = 2

for iter_num in range(epoch):
    print("iteration = %d" % (iter_num + 1))
    for data, label in zip(X_train, y_train):
        targets = np.full(10, 0.01)
        targets[label] = 0.99

        inputs = np.array(data, dtype=float)  # / 255.0 * 0.99) + 0.01
        inputs = (inputs / 255.0 * 0.99) + 0.01

        myNN.fit(inputs, targets)
        # myNN.fit_by_GD(inputs, targets)

    # myNN.updateWeight(len(y_train))


print("=== done ===")
print()

# test
print("=== test === ")

scoredata = []

# test data
for data, label in zip(X_test, y_test):

    target_o = label

    inputs = np.array(data, dtype=float)  # / 255.0 * 0.99) + 0.01
    inputs = (inputs / 255.0 * 0.99) + 0.01

    # emulation
    res = myNN.predict(inputs)
    output = np.argmax(res)
    if (target_o == output):
        # print("correct, ", end = " ")
        scoredata.append(1)
    else:
        # print("NG %d : (%d vs %d) " %(idx, target_o, output) ,end = " ")
        scoredata.append(0)

print()
print("== done (%d samples)==" % (len(scoredata)))
# print(scoredata)
print("performance = ", sum(scoredata) / len(scoredata))
