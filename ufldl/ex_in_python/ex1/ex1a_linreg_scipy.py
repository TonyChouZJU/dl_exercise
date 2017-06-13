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
print 'X shape:', train_X.shape, ' y shape:',train_y.shape


m, n = train_X.shape

def cost_function(theta, X, y): 
    squared_errors = (X.dot(theta) - y) ** 2
    J = 0.5 * squared_errors.sum()
    print 'IN cost:',' X.shape:',X.shape,' y.shape:',y.shape, ' theta:',theta.shape, ' errors.shape:', squared_errors.shape, ' cost.shape:', J.shape
    return J

def gradient(theta, X, y):
    errors = X.dot(theta) - y
    theta_delta = errors.dot(X)
    print 'In gradient:',  'X.shape:',X.shape,' y.shape:',y.shape, ' theta:',theta.shape, ' errors.shape:', errors.shape, ' theta_delta.shape:', theta_delta.shape
    return theta_delta


print 'cost:', cost_function(np.random.rand(n), train_X, train_y)

def print_cost(x):
    print 'theta:',x

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
    #callback=lambda x: print_cost(x),
)
print 'J_history:', J_history
t1 = time.time()

print('Optimization took {s} seconds'.format(s=t1 - t0))
optimal_theta = res.x

plt.plot(J_history, marker='o')
plt.xlabel('Iterations')
plt.ylabel('J(theta)')
plt.show()

## look at the root mean squared error
for df, (X, y) in (('train', (train_X, train_y)),('test', (test_X, test_y))):
    actual_prices = y
    predicted_prices = X.dot(optimal_theta)
    print('RMS {dataset} error: {error}'.format(dataset=df,
                                                error=np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))         )
    )


## plotting the test data
pred_prices = np.dot(test_X, optimal_theta)
#pred_prices = linear_regression(optimal_theta, test_X, test_y)
print 'pred_prices shape:',pred_prices.shape

print 'test_y.size:', test_y.size
print 'pred_y.size:', pred_prices.size
plt.figure(figsize=(10, 8))
plt.scatter(np.arange(test_y.size), sorted(test_y), c='r', edgecolor='None', alpha=0.5, label='Actual')
plt.scatter(np.arange(test_y.size), sorted(pred_prices), c='b', edgecolor='None', alpha=0.5, label='Predicted')
plt.legend(loc='upper left')
plt.title("Predicted vs. Actual House Price")
plt.ylabel('Price ($1000s)')
plt.xlabel('House #')
plt.show()
