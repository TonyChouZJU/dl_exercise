import numpy as np
import scipy.optimize
import time
import matplotlib.pylab as plt

def linear_regression(theta,X,y):
	y_preds = X.dot(theta)
	return y_preds

def cost_function(theta, X, y):
	y_preds = linear_regression(theta, X, y)
	loss = np.sum(np.square(y_preds-y))/(2*y.shape[0])
	return loss

def gradient(theta, X, y):
	y_preds = linear_regression(theta, X, y)
	errors = y_preds-y

	theta_grad = errors.dot(X)/ y.shape[0]
	return theta_grad


def optimizer(theta_init, X, y, iters=200):
	theta = theta_init
	c = 0.01
	#for i in range(X.shape[1]):
	for i in range(iters):
		X_i = X[i%X.shape[0], :].reshape((-1,theta.shape[0]))
		y_i = y[i%X.shape[0], :].reshape((-1,1))
		loss = cost_function(theta, X_i, y_i)
		theta_grad = gradient(theta, X_i, y_i)
		theta = theta -  c * theta_grad.transpose()
		print loss,'\t\t\t', theta.flatten()
	return theta

def load_data(input_file='housing.data'):
	np_data= np.zeros((0,14),np.float32)
	#np_data= np.zeros((0,3),np.float32)
	with open(input_file, 'rb') as f:
		for line in f:
			this_line = [float(p) for p in line.strip().split()]
			np_data = np.vstack([np_data, np.array(this_line)])
	
	np_data = np.hstack([np.ones((np_data.shape[0], 1)), np_data])
	return np_data

dataset = load_data(input_file='housing.data')
#dataset = load_data(input_file='test.data')
print 'dataset:', dataset
data_size = dataset.shape[0]
train_ratio = 1
train_size= int(data_size * train_ratio)

train_x = dataset[:train_size, :-1]
train_y = dataset[:train_size, -1:]

test_x = dataset[train_size:, :-1]
test_y = dataset[train_size:, -1:]

theta_init = np.random.rand(train_x.shape[1], 1)

#print 'loss:\t\t theta:'
#theta_final = optimizer(theta_init, train_x, train_y, iters=200)

#print 'theta init:',  theta_init.flatten()
#print 'theta final:', theta_final.flatten()
			
J_history = []

t0 = time.time()
#res = scipy.optimize.minimize(
#    fun=cost_function,
#    x0=np.random.rand(train_x.shape[1], 1 ),
#    args=(train_x, train_y),
#    method='bfgs',
#    jac=gradient,
#    options={'maxiter': 200, 'disp': True},
#    callback=lambda x: J_history.append(cost_function(x, train_x, train_y)),
#)
res = scipy.optimize.minimize(
    fun=cost_function,
    x0=np.random.rand(train_x.shape[1] ),
    args=(train_x, train_y),
    method='bfgs',
    jac=gradient,
    options={'maxiter': 200, 'disp': True},
)
t1 = time.time()

print('Optimization took {s} seconds'.format(s=t1 - t0))
optimal_theta = res.x
print optimal_theta
print 'final loss:', cost_function(optimal_theta, train_x, train_y)
plt.plot(J_history, marker='o')
plt.xlabel('Iterations')
plt.ylabel('J(theta)')
plt.show()
