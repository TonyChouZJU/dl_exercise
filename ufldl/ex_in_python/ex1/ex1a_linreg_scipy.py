import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import time

data_original = np.loadtxt('./housing.data')
data = np.insert(data_original, 0, 1, axis=1)
print 'origin data:', data
np.random.shuffle(data)


train_X = data[:400, :-1]
train_y = data[:400, -1]

test_X = data[400:, :-1]
test_y = data[400:, -1]

m, n = train_X.shape

def cost_function(theta, X, y): 
    squared_errors = (X.dot(theta) - y) ** 2
    J = 0.5 * squared_errors.sum()
    return J

def gradient(theta, X, y):
    errors = X.dot(theta) - y
    return errors.dot(X)


print 'cost:', cost_function(np.random.rand(n), train_X, train_y)

assert False

J_history = []

t0 = time.time()
res = scipy.optimize.minimize(
    fun=cost_function,
    x0=np.random.rand(n),
    args=(train_X, train_y),
    method='bfgs',
    jac=gradient,
    options={'maxiter': 200, 'disp': True},
    callback=lambda x: J_history.append(cost_function(x, train_X, train_y)),
)
t1 = time.time()

print('Optimization took {s} seconds'.format(s=t1 - t0))
optimal_theta = res.x

plt.plot(J_history, marker='o')
plt.xlabel('Iterations')
plt.ylabel('J(theta)')
plt.show()
