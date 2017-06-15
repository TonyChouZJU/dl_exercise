import matplotlib.pylab as plt
from sklearn import datasets, svm, metrics

import numpy as np
import scipy.optimize
import time

from sklearn import preprocessing, linear_model
import scipy.special

import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#define the minimium value of a float32
FLT_POSITIVE_MIN = np.finfo(np.float32).tiny
FLT_POSITIVE_MAX = np.finfo(np.float32).max

#there are many different implementation of sigmoid
#Here we prefer to use the scipy implementation
#expit(x) = 1/(1+_exp(-x))
def logistic_regression(theta, X, y):
    y_preds = scipy.special.expit( X.dot(theta) )
    #the following is a little slower than scipy.special.expit
    #y_preds = 1.0 /(1.0 +  np.exp(-X.dot(theta)) )
    #y_preds = 1.0 /(1.0 +  math.e**(-X.dot(theta)))
    return y_preds

#Here we use cross entropy loss, we want to minimize the loss
#J_theta = - sum_i ( y_i * log (y_preds) + (1- y_i) * log( 1 - y_preds))
#The loss is depend on the probability computed from logistic_regression
def cost_function(theta, X, y):
    y_preds = logistic_regression(theta, X, y)

    #Important:be careful about the overflow 
    #the sign +/- make no sense, as we want to minimize the cost to zero
    Hy = - ( y.dot( np.log(np.maximum(y_preds, FLT_POSITIVE_MIN)) ) + (1.0- y).dot(np.log( np.maximum(1.0 - y_preds, FLT_POSITIVE_MIN) )) )

    return Hy

#It is the same as linear regression except for h_theta(x) = sigmoid(x)
def gradient(theta, X, y):
    y_preds = logistic_regression(theta, X, y)
    errors = y_preds -y
    theta_grad = (errors.T).dot(X) /y.shape[0]
    return theta_grad

def optimizer(theta_init, X, y, iters=200, converge_change=-1e-60):
        loss_history = []
	theta = theta_init
        n_samples = len(X)
	lr = 0.1
        loss = cost_function(theta_init, X, y)
        change_loss = 1.
        batch_size = 20

	for i in range(iters):
                old_loss = loss 
                if i == int(iters*0.5):
                    lr = lr*0.1
                elif i == int(iters*0.7):
                    lr = lr*0.1
                elif i == int(iters*0.9):
                    lr = lr*0.1

                sample_idx = np.arange(n_samples)
                np.random.shuffle(sample_idx)

                #rand_batch = np.random.randint(n_samples-2)
                rand_batch_idx =  sample_idx[:batch_size]
                X_i = X[rand_batch_idx, :].reshape((-1,theta.shape[0]))
                y_i = y[rand_batch_idx]
		#X_i = X
                #y_i = y
		theta_grad = gradient(theta, X_i, y_i)
		theta = theta - lr * theta_grad.transpose()
		loss = cost_function(theta, X_i, y_i)
                loss_history.append(loss)
                logging.info("iter: {:8d}\n\t\t\tloss: {:6f}".format(i, loss))
                logging.info( '\t\t\tRMS error: {error}'.format( error=np.sqrt(np.mean( (logistic_regression(theta, X_i, y_i) - y_i) ** 2))) )

                change_loss = old_loss - loss
                #if change_loss <= converge_change:
                #    break
	return theta, loss_history

def show_J_history(J_history):
    #plt.plot(J_history, marker='o')
    plt.plot(J_history)
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
    plt.title("Predicted vs. Actual Digits")
    plt.ylabel('Digits (0 or 1)')
    plt.xlabel('samples #')
#    plt.show()
    plt.savefig(sv_name)

def logistic_regression_sklearn(train_x, train_y, test_x, test_y, theta_init,  sv_name, iters=200):
    reg = linear_model.LogisticRegression(penalty='l2', fit_intercept=False)
    reg.fit(train_x, train_y)

    theta = reg.coef_.flatten()
    #theta = np.hstack([reg.intercept_, reg.coef_])
    #train_x = np.hstack([np.ones((train_x.shape[0], 1)), train_x])
    #test_x = np.hstack([np.ones((test_x.shape[0], 1)), test_x])

    logging.info('reg coef:{}, shape;{}, intercept_:{}'.format(reg.coef_, reg.coef_.shape, reg.intercept_))

    pred_y = logistic_regression(theta, test_x, test_y)
    show_results(test_y, pred_y, sv_name)
    #analysis_error(train_x, train_y, test_x, test_y, theta)
    return theta
    
     

def gradient_descend(train_x, train_y, test_x, test_y, theta_init, sv_name,  iters=200, converge_change=-1e-60):
    theta_final, J_history = optimizer(theta_init, train_x, train_y, iters, converge_change)
    			
    pred_prices = logistic_regression(theta_final, test_x, test_y)

    show_J_history(J_history)
    show_results(test_y, pred_prices, sv_name)
    return theta_final

def split_digits_data(dgts, train_ratio=0.8):
    n_samples = len(dgts.images)
    train_size = int(n_samples * train_ratio)

    all_x = dgts.images.reshape((n_samples, -1)).astype(np.float32)
    all_y = dgts.target.astype(np.float32)

    sample_idx = np.arange(n_samples)
    np.random.shuffle(sample_idx)

    all_x = all_x[sample_idx]
    all_y = all_y[sample_idx]

    train_x = all_x[:train_size, :]
    train_y = all_y[:train_size]

    test_x = all_x[train_size:, :]
    test_y = all_y[train_size:]

    return train_x, train_y, test_x, test_y




if __name__=='__main__':
    # The digits dataset
    digits = datasets.load_digits(n_class=2)

    orig_train_x, train_y, orig_test_x, test_y = split_digits_data(digits)
    #theta_init = np.random.rand(orig_train_x.shape[1])
    theta_init = np.zeros(orig_train_x.shape[1]) + 0.0001
    print orig_test_x, test_y
    #optimal_theta_1 = gradient_descend(orig_train_x, train_y, orig_test_x, test_y, theta_init, sv_name='./results/gd_logistic.png', iters=50)
    optimal_theta_1 = logistic_regression_sklearn(orig_train_x, train_y, orig_test_x, test_y, theta_init, sv_name='./results/sk_logistic_regression_l2', iters=50)

    prob_y = logistic_regression(optimal_theta_1, orig_test_x, test_y)
    pred_y = np.where(prob_y>=0.5, 1, 0) == test_y

    logging.info('TP:{} Vs Total:{},  Accuracy: {}'.format(np.sum(pred_y), len(pred_y), np.sum(pred_y)*1. /len(pred_y)))
