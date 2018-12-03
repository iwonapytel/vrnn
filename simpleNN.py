import numpy as np

data = np.load('lab03-data-mnist.npz')
Xtr,Ytr,Xte,Yte = data['Xtr'],data['Ytr'],data['Xte'],data['Yte']
np.random.seed(1000)
weights1 = 2 * np.random.random((784, 100)) - 1
weights2 = 2 * np.random.random((100, 10)) - 1

# TODO: implement a function which calculates the sigmoid / derivative of the sigmoid function
def sigmoid(x, deriv = False):
    f = 1 / (1 + np.exp(-x))
    if(deriv==True):
        return f * (1 - f)
    return f

def accuracy(pred, gt):
    return np.sum(pred == gt) / pred.shape[0]

def forward_pass(X):
    layer1 = sigmoid(np.dot(X, weights1))
    layer2 = sigmoid(np.dot(layer1, weights2))
    return layer1, layer2

lr =  0.1 # TODO: 0.1 / 0.01
for i in range(500):
    # Feed forward
    layer1, layer2 = forward_pass(Xtr)

    # Back propagation using gradient descent. Loss=1/2 L2.
    layer2_err = Ytr - layer2
    layer2_delta = layer2_err * sigmoid(layer2, deriv=True)

    layer1_err = layer2_delta.dot(weights2.T)
    layer1_delta = layer1_err * sigmoid(layer1, deriv=True)

    weights1 += lr * Xtr.T.dot(layer1_delta)
    weights2 += lr * layer1.T.dot(layer2_delta)

    if i % 10 == 0:
        layer1, layer2 = forward_pass(Xtr)
        print("Accuracy [TRAIN SET]:",accuracy( layer2.argmax(axis=1), Ytr.argmax(axis=1)))

        layer1, layer2 = forward_pass(Xte)
        print("Accuracy [TEST SET]:",accuracy( layer2.argmax(axis=1), Yte.argmax(axis=1)))
