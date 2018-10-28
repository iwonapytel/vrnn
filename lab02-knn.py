# **** Lab. 2 ****
# CIFAR10: implement (preferred) or apply (allowed) k-Nearest Neighbor {1,3,5,7} with L1 and L2 metrics and report accuracy for validation and test sets, along with confusion matrices for the test set.
# Useful: sklearn.neighbors.KNeighborsClassifier, sklearn.metrics.accuracy_score, sklearn.metrics.confusion_matrix .


from cifar10 import load_CIFAR10
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def load_data():
    Xtr, Ytr, Xte, Yte = load_CIFAR10('cifar-10-batches-py')
    Xtr = np.array(Xtr,dtype=np.int32)
    Ytr = np.array(Ytr,dtype=np.int32)
    Xte = np.array(Xte,dtype=np.int32)
    Yte = np.array(Yte,dtype=np.int32)
    Xva = Xtr[40001:,:]
    Yva = Ytr[40001:]
    Xtr = Xtr[:40000,:]
    Ytr = Ytr[:40000]
    indices = range(0,Xte.shape[0],200)
    return Xtr, Ytr, Xva[indices,:], Yva[indices], Xte[indices,:], Yte[indices]


Xtr, Ytr, Xva, Yva, Xte, Yte = load_data()
ntest = Xte.shape[0]

class NearestNeighbors:
    def __init__(Xtr, Ytr):
        self.X = Xtr
        self.Y = Ytr

    def predict(Xte, metric):
        tr_dim = Xte.shape[0]
        pred = np.zeros(tr_dim, dtype=self.Y.dtype)
        for i in xrange(tr_dim):
            distances = metric(self.X, Xte[i,:])
            nearest = np.argmin(distances)
            pred[i] = self.Y[nearest]
        return pred

l1 = lambda x, y : np.sum(np.abs(x-y), axis=1)
l2 = lambda x, y : np.sqrt(np.sum(np.square(x-y), axis=1))

def accuracy(y_true, y_pred):
    correct = (y_true == y_pred)
    return correct.sum() / float(correct.size())

def confusion_matrix_(y_true, y_pred):
    labels = np.unique(y_true)
    matrix = numpy.zeros(len(labels), len(labels))
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return matrix

# NN - please do not use any library
print "Nearest Neighbor"
nn = NearestNeighbors(Xtr, Ytr)
pred_nn_va_l1 = nn.predict(Xva, l1)
pred_nn_va_l2 = nn.predict(Xva, l2)
pred_nn_te_l1 = nn.predict(Xte, l1)
pred_nn_te_l2 = nn.predict(Xte, l2)

print "Accuracy L1:", accuracy(Yva, pred_nn_va_l1), accuracy(Yte, pred_nn_te_l1)
print confusion_matrix_(Yva, pred_nn_te_l1)
print "Accuracy L2:", accuracy(Yva, pred_nn_va_l2), accuracy(Yte, pred_nn_te_l2)
print confusion_matrix_(Yte, pred_nn_te_l2)

# k-NN - implement (preferred) or apply from a library (allowed)
for k in [1,3,5,7]:
    print "k-NearestNeighbors for k =",k
    knn_l1 = KNeighborsClassifier(n_neighbors=k, p=1)
    knn_l1.fit(Xtr, Ytr)
    pred_knn_va_l1 = knn_l1.predict(Xva)
    pred_knn_te_l1 = knn_l1.predict(Xte)
    accuracy_on_val_set_l1 = accuracy_score(Yva, pred_knn_va_l1)
    accuracy_on_test_set_l1 = accuracy_score(Yte, pred_knn_te_l1)

    knn_l2 = KNeighborsClassifier(n_neighbors=k, p=2)
    knn_l2.fit(Xtr, Ytr)
    pred_knn_va_l2 = knn_l2.predict(Xva)
    pred_knn_te_l2 = knn_l2.predict(Xte)
    accuracy_on_val_set_l2 = accuracy_score(Yva, pred_knn_va_l2)
    accuracy_on_test_set_l2 = accuracy_score(Yte, pred_knn_te_l2)

    print "Accuracy L1:",  accuracy_on_val_set_l1, accuracy_on_test_set_l1
    print confusion_matrix(Yte, pred_knn_te_l1)
    print "L1=<accuracy_on_val_set>/<accuracy_on_test_set>", \
        accuracy_on_val_set_l1/accuracy_on_test_set_l1

    print "Accuracy L2:", accuracy_on_val_set_l2, accuracy_on_test_set_l2
    print confusion_matrix(Yte, pred_knn_te_l2)
    print "L2=<accuracy_on_val_set>/<accuracy_on_test_set>", \
        accuracy_on_val_set_l2/accuracy_on_test_set_l2
