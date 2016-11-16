import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from scipy.spatial.distance import pdist, squareform

start_time = time.time()

# compute radial base kernel, given the hyperparameter t
def RBF_kernel(train_data, RBF_t):
    pairwise_sq_dists = squareform(pdist(train_data, 'sqeuclidean'))
    kernel = scipy.exp(- RBF_t * (pairwise_sq_dists))
    return kernel
    
# definition of objective function of the dual problem (maximization problem)
def objective_function(train_data, RBF_t, beta):
    ker = RBF_kernel(train_data, RBF_t)
    objective_value = -0.5 * np.dot(np.dot(beta.T,ker),beta) + sum(abs(beta))
    return objective_value

#directory & filenames
directory = 'giesen14_data/'
a1a_filename = 'a1a'
a2a_filename = 'a2a'
a3a_filename = 'a3a'
a4a_filename = 'a4a'
ionosphere = 'ionosphere_scale'
txt = '.txt'
tst = '.t'

#constants
c = 0.1
n = 100
n_features = 123

#load svm dataset from LIBSVM site
x_train, y_train = load_svmlight_file(directory + a1a_filename + txt, n_features = n_features)
x_test, y_test= load_svmlight_file(directory + a1a_filename + tst, n_features = n_features)

#dense matrix -> sparse matrix conversion
x_train = np.array(x_train.todense())
x_test = np.array(x_test.todense())

n_sample = y_train.shape[0]

#set t in log space from 2^-10 to 2^10
t = np.logspace(-10, 10, num = n, base = 2, dtype = 'float64')[:,np.newaxis]

#initialize the optimum matrix for each t available
obj = np.zeros(shape = (n, 1))

for i in range(n):
    
    beta = np.zeros(shape = (n_sample,1))
    
    # kernel = radial basis function ; c = 0.1
    clf = SVC(kernel = 'rbf', gamma = t[i], C = c)
    clf.fit(x_train, y_train)
  
    #construct beta (beta[i] = y[i] * alpha[i])
    #get the index for each non-shrunk parameter and assign to beta array
    #the non-assigned beta indexes are the shrunk parameters
    for j in range(clf.support_.shape[0]):
        idx = clf.support_[j]
        beta[idx] = clf.dual_coef_.T[j]
    
    obj[i] = objective_function(x_train, t[i], beta)
    
#plotting graph
plt.plot(t, obj)
plt.title('giesen14 experiment: a1a')
plt.ylabel('objective value')
plt.xlabel('kernel parameter t')
plt.xscale('log', basex = 2)
plt.axis([t[0] / 2, t[n-1], min(obj) - 5, max(obj) + 5])
plt.show()

elapsed_time = time.time() - start_time
print elapsed_time