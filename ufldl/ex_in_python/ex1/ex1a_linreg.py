import numpy as np
import scipy.optimize
import time
import matplotlib.pylab as plt

from sklearn import preprocessing

import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class featureNormlize():
    def __init__(self, X, method='stand'):
        self._method = method
        self._scaler = self.initScaler(X)

    def initScaler(self, X):
        if self._method=='stand':
            return preprocessing.StandardScaler().fit(X)
        #default min and max is 0 and 1
        #If MinMaxScaler is given an explicit feature_range=(min, max) the full formula is:
        #X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        #X_scaled = X_std / (max - min) + min
        elif self._method=='minmax':
            return preprocessing.MinMaxScaler()
        elif self._method=='maxabs':
            return preprocessing.MaxAbsScaler()
        elif self._method=='l2':
            return preprocessing.Normalizer.fit(X, norm='l2')
        elif self._method=='l1':
            return preprocessing.Normalizer.fit(X, norm='l1')
        else:
            return None
    def transform(self, x):
        if self._scaler != 'None':
            return self._scaler.transform(x)
        else:
            return None

def linear_regression(theta,X,y):
	y_preds = X.dot(theta)
	return y_preds

def cost_function(theta, X, y):
	y_preds = linear_regression(theta, X, y)
	loss = np.sum(np.square(y_preds-y))/(2*y.shape[0])
	#loss = np.sum(np.square(y_preds-y))/2
	return loss

def gradient(theta, X, y):
	y_preds = linear_regression(theta, X, y)
	errors = y_preds-y

	theta_grad = (errors.T).dot(X)/ y.shape[0]
	return theta_grad


def optimizer(theta_init, X, y, iters=200):
        loss_history = []
	theta = theta_init
	c = 0.1
	#for i in range(X.shape[1]):
	for i in range(iters):
                if i == int(iters*0.5):
                    c = c*0.1
                elif i == int(iters*0.7):
                    c = c*0.1
                elif i == int(iters*0.9):
                    c = c*0.1


		#X_i = X[i%X.shape[0], :].reshape((-1,theta.shape[0]))
		X_i = X
		#y_i = y[i%X.shape[0]].reshape((-1,1))
                #y_i = y[i%X.shape[0]].reshape(1)
                y_i = y
		loss = cost_function(theta, X_i, y_i)
		theta_grad = gradient(theta, X_i, y_i)
		theta = theta -  c * theta_grad.transpose()
                loss_history.append(loss)
                #logging.info("iter: {:8d}\n\t\t\tloss: {:6f}".format(i, loss))
                #logging.info( '\t\t\tRMS error: {error}'.format( error=np.sqrt(np.mean( (linear_regression(theta, X_i, y_i) - y_i) ** 2))) )
	return theta, loss_history

def load_data(input_file='housing.data'):
	np_data= np.zeros((0,14),np.float32)
	#np_data= np.zeros((0,3),np.float32)
	with open(input_file, 'rb') as f:
		for line in f:
			this_line = [float(p) for p in line.strip().split()]
			np_data = np.vstack([np_data, np.array(this_line)])
	
	#np_data = np.hstack([np.ones((np_data.shape[0], 1)), np_data])
	return np_data

def split_data(dataset, train_ratio=0.8):
    data_size = dataset.shape[0]
    train_size= int(data_size * train_ratio)
    
    train_x = dataset[:train_size, :-1]
    train_y = dataset[:train_size, -1]
    
    test_x = dataset[train_size:, :-1]
    test_y = dataset[train_size:, -1]
    return train_x, train_y, test_x, test_y


def show_J_history(J_history):
    plt.plot(J_history, marker='o')
    plt.title("Theta History")
    plt.xlabel('Iterations')
    plt.ylabel('J(theta)')
    plt.show()

def show_results(test_y, y_preds, sv_name):
    plt.figure(figsize=(10, 8))
    plt.scatter(np.arange(test_y.size), sorted(test_y), c='r', edgecolor='None', alpha=0.5, label='Actual')
    plt.scatter(np.arange(test_y.size), sorted(y_preds), c='b', edgecolor='None', alpha=0.5, label='Predicted')
    range_y = int(max(test_y)-min(test_y))
    plt.legend(loc='upper left')
    plt.ylim([min(test_y)-0.1*range_y, max(test_y)+0.1*range_y])
    plt.title("Predicted vs. Actual House Price")
    plt.ylabel('Price ($1000s)')
    plt.xlabel('House #')
#    plt.show()
    plt.savefig(sv_name)

def gradient_descend(train_x, train_y, test_x, test_y, theta_init, sv_name,  iters=200):
    theta_final, J_history = optimizer(theta_init, train_x, train_y, iters)
    			
    pred_prices = linear_regression(theta_final, test_x, test_y)

    #show_J_history(J_history)
    show_results(test_y, pred_prices, sv_name)
    return theta_final
    
def scipy_optimize(train_x, train_y, test_x, test_y,  theta_init, sv_name, iters=200, _method='bfgs'):
    J_history = []
    
    t0 = time.time()
    res = scipy.optimize.minimize(
        fun=cost_function,
        x0=theta_init,
        args=(train_x, train_y),
        method=_method,
        jac=gradient,
        options={'maxiter': iters, 'disp': True},
        callback=lambda x: J_history.append(cost_function(x, train_x, train_y)),
    )
    t1 = time.time()
    
    optimal_theta = res.x

    pred_prices = linear_regression(optimal_theta, test_x, test_y)
    #show_J_history(J_history)
    show_results(test_y, pred_prices, sv_name)
    return optimal_theta

#y = theta*X
#theta = (XTX)-1*XT*y
def normal_equation(train_x, train_y, test_x, test_y, sv_name):
    xt_x = (train_x.T).dot(train_x)
    xt_x_inv = np.linalg.inv(xt_x)
    theta = xt_x_inv.dot(train_x.T).dot(train_y)

    pred_y = linear_regression(theta, test_x, test_y)
    show_results(test_y, pred_y, sv_name)
    return theta

if __name__ == '__main__':
    dataset = load_data(input_file='housing.data')
    #print 'dataset:', dataset
    np.random.shuffle(dataset)
    
    
    orig_train_x, train_y, orig_test_x, test_y = split_data(dataset)
    
    normlizer = featureNormlize(orig_train_x,'stand')

    train_x = normlizer.transform(orig_train_x)
    train_x = np.hstack([np.ones((train_x.shape[0], 1)), train_x])

    orig_train_x = np.hstack([np.ones((orig_train_x.shape[0], 1)), orig_train_x])
    
    test_x = normlizer.transform(orig_test_x)
    test_x = np.hstack([np.ones((test_x.shape[0], 1)), test_x])

    orig_test_x = np.hstack([np.ones((orig_test_x.shape[0], 1)), orig_test_x])
    
    theta_init = np.random.rand(train_x.shape[1])

    
    #hand write gradient descend
    optimal_theta_1 = gradient_descend(train_x, train_y, test_x, test_y, theta_init, sv_name='./results/gt_vs_pred_1.png', iters=200)
    optimal_theta_2 = scipy_optimize(train_x, train_y, test_x, test_y, theta_init, sv_name='./results/gt_vs_pred_2.png',iters=200)
    optimal_theta_3 = normal_equation(train_x, train_y, test_x, test_y, sv_name='./results/gt_vs_pred_3.png')

    logging.info('Loss 1:{}'.format(cost_function(optimal_theta_1, train_x, train_y)))
    logging.info('Loss 2:{}'.format(cost_function(optimal_theta_2, train_x, train_y)))
    logging.info('Loss 3:{}'.format(cost_function(optimal_theta_3, train_x, train_y)))

    optimal_theta_4 = gradient_descend(orig_train_x, train_y, orig_test_x, test_y, theta_init, sv_name='./results/gt_vs_pred_4.png', iters=200)
    optimal_theta_5 = scipy_optimize(orig_train_x, train_y, orig_test_x, test_y, theta_init, sv_name='./results/gt_vs_pred_5.png',iters=200)
    optimal_theta_6 = normal_equation(orig_train_x, train_y, orig_test_x, test_y, sv_name='./results/gt_vs_pred_6.png')

    logging.info('Loss 4:{}'.format(cost_function(optimal_theta_4, orig_train_x, train_y)))
    logging.info('Loss 5:{}'.format(cost_function(optimal_theta_5, orig_train_x, train_y)))
    logging.info('Loss 6:{}'.format(cost_function(optimal_theta_6, orig_train_x, train_y)))
    
    optimal_theta_7 = scipy_optimize(orig_train_x, train_y, orig_test_x, test_y, theta_init, sv_name='./results/gt_vs_pred_7.png',iters=200, _method='Powell')
    logging.info('Loss 7:{}'.format(cost_function(optimal_theta_7, orig_train_x, train_y)))
    ###################
    ## Error and Plots ##
    ###################
    
    ## look at the root mean squared error
    for df, (X, y) in (('train', (train_x, train_y)),('test', (test_x, test_y))):
        actual_prices = y
        predicted_prices = X.dot(optimal_theta_2)
        logging.info('RMS {dataset} error: {error}'.format(dataset=df,
                                                    error=np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))         )
        )
