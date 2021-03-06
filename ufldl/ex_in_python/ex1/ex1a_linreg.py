import numpy as np
import scipy.optimize
import time
import matplotlib.pylab as plt

from sklearn import preprocessing, linear_model

import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#class featureNormlize():
#This could behave badly if the individual feature do not more or less like standard normally distributed data
class featureStandlizer():
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
            return preprocessing.Normalizer(norm='l2')
        elif self._method=='l1':
            return preprocessing.Normalizer(norm='l1')
        elif self._method=='max':
            return preprocessing.Normalizer(norm='max')
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

def newton(theta, X, y):
	#hession matrix
	h = (X.T).dot(X)
	g = gradient(theta, X, y)
	d = g/np.diag(h)
	return d

def gradient(theta, X, y):
	y_preds = linear_regression(theta, X, y)
	errors = y_preds-y

	theta_grad = (errors.T).dot(X)/ y.shape[0]
	return theta_grad

def newton_optimizer(theta_init, X, y, iters=10):
        loss_history = []
	theta = theta_init
	for i in range(iters):

		#X_i = X[i%X.shape[0], :].reshape((-1,theta.shape[0]))
		X_i = X
		#y_i = y[i%X.shape[0]].reshape((-1,1))
                #y_i = y[i%X.shape[0]].reshape(1)
                y_i = y
		loss = cost_function(theta, X_i, y_i)
		n_grad = newton(theta, X_i, y_i)
		theta = theta - n_grad  
                loss_history.append(loss)
	return theta, loss_history

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
	return theta, loss_history

def load_data(input_file='housing.data'):
	np_data= np.zeros((0,14),np.float32)
	with open(input_file, 'rb') as f:
		for line in f:
			this_line = [float(p) for p in line.strip().split()]
			np_data = np.vstack([np_data, np.array(this_line)])
	
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
    #plt.show()
    plt.savefig(sv_name)

def newton_descend(train_x, train_y, test_x, test_y, theta_init, sv_name,  iters=10):
    logging.info('==============================Using the newton method by myself ')
    theta_final, J_history = newton_optimizer(theta_init, train_x, train_y, iters)
    			
    pred_prices = linear_regression(theta_final, test_x, test_y)

    #show_J_history(J_history)
    show_results(test_y, pred_prices, sv_name)
    analysis_error(train_x, train_y, test_x, test_y, theta_final)
    return theta_final

def gradient_descend(train_x, train_y, test_x, test_y, theta_init, sv_name,  iters=200):
    logging.info('==============================Using the gradient_descend by myself ')
    theta_final, J_history = optimizer(theta_init, train_x, train_y, iters)
    			
    pred_prices = linear_regression(theta_final, test_x, test_y)

    #show_J_history(J_history)
    show_results(test_y, pred_prices, sv_name)
    analysis_error(train_x, train_y, test_x, test_y, theta_final)
    return theta_final
    
def scipy_optimize(train_x, train_y, test_x, test_y,  theta_init, sv_name, iters=200, _method='bfgs'):
    logging.info('==============================Using minimize function in scipy kit')
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
    analysis_error(train_x, train_y, test_x, test_y, optimal_theta)
    return optimal_theta

#y = theta*X
#theta = (XTX)-1*XT*y
def normal_equation(train_x, train_y, test_x, test_y, sv_name):
    logging.info('==============================Using a closed-form solution')
    xt_x = (train_x.T).dot(train_x)
    xt_x_inv = np.linalg.inv(xt_x)
    theta = xt_x_inv.dot(train_x.T).dot(train_y)

    pred_y = linear_regression(theta, test_x, test_y)
    show_results(test_y, pred_y, sv_name)
    analysis_error(train_x, train_y, test_x, test_y, theta)
    return theta

def linear_regression_sklearn(train_x, train_y, test_x, test_y, theta_init,  sv_name):
    logging.info('==============================Using sklearn LinearRegression, minimize the residual sum of squares')
    #as the data is not centered, we can use two method to get the intercept
    #a)fit_intercpt = True, we can have an extra paramaters intercept_, 
    #b)add 1 into the origin data, and set fit_intercept=False
    #reg = linear_model.LinearRegression(fit_intercept=False)
    reg = linear_model.LinearRegression(fit_intercept=True)
    reg.fit(train_x, train_y)

    theta = np.hstack([reg.intercept_, reg.coef_])
    train_x = np.hstack([np.ones((train_x.shape[0], 1)), train_x])
    test_x = np.hstack([np.ones((test_x.shape[0], 1)), test_x])

    logging.info('reg coef:{}, shape;{}, intercept_:{}'.format(reg.coef_, reg.coef_.shape, reg.intercept_))

    pred_y = linear_regression(theta, test_x, test_y)
    #pred_y = reg.predict(test_x)
    show_results(test_y, pred_y, sv_name)
    analysis_error(train_x, train_y, test_x, test_y, theta)
    return theta

#Ridge regression: linear least squares with l2 normalization
def ridge_regression(theta,X,y):
    pass

def ridge_regression_sklearn(train_x, train_y, test_x, test_y, theta_init,  sv_name):
    logging.info('==============================Using ridge regression which address the problem of ordinary least square by imporsing a penalty on the size of coeffienct')
    reg = linear_model.Ridge(alpha=1.0, fit_intercept=True)
    reg.fit(train_x, train_y)

    theta = np.hstack([reg.intercept_, reg.coef_])
    train_x = np.hstack([np.ones((train_x.shape[0], 1)), train_x])
    test_x = np.hstack([np.ones((test_x.shape[0], 1)), test_x])

    logging.info('reg coef:{}, shape;{}, intercept_:{}'.format(reg.coef_, reg.coef_.shape, reg.intercept_))

    pred_y = linear_regression(theta, test_x, test_y)
    #pred_y = reg.predict(test_x)
    show_results(test_y, pred_y, sv_name)
    analysis_error(train_x, train_y, test_x, test_y, theta)
    return theta

def ridge_regression_cv_sklearn(train_x, train_y, test_x, test_y, theta_init,  sv_name):
    logging.info('==============================Using ridge regression with built-in cross-validation of the alpha paramters')
    reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10], fit_intercept=True)
    reg.fit(train_x, train_y)

    theta = np.hstack([reg.intercept_, reg.coef_])
    train_x = np.hstack([np.ones((train_x.shape[0], 1)), train_x])
    test_x = np.hstack([np.ones((test_x.shape[0], 1)), test_x])

    logging.info('reg coef:{}, shape;{}, intercept_:{}'.format(reg.coef_, reg.coef_.shape, reg.intercept_))
    logging.info('reg alpha:{}'.format(reg.alpha_))

    pred_y = linear_regression(theta, test_x, test_y)
    #pred_y = reg.predict(test_x)
    show_results(test_y, pred_y, sv_name)
    analysis_error(train_x, train_y, test_x, test_y, theta)
    return theta


def analysis_error(train_x, train_y, test_x, test_y, optimal_theta):

    ## look at the root mean squared error
    for df, (X, y) in (('train', (train_x, train_y)),('test', (test_x, test_y))):
        actual_prices = y
        predicted_prices = X.dot(optimal_theta)
        mean_squared_error = np.mean((predicted_prices - actual_prices) ** 2)
        logging.info('Mean squared error {dataset} error:\t\t\t\t\t {error:>8}'.format(dataset=df,
                                                    error=mean_squared_error    ))
        root_mean_squared_error = np.sqrt(mean_squared_error) 
        logging.info('Root mean squared error {dataset} error:\t\t\t\t\t {error:>8}'.format(dataset=df,
                                                    error=root_mean_squared_error    ))
        #residual sum of squares 
        RSS = np.sum( (predicted_prices - actual_prices) ** 2)
        #Eplained sum of squares
        ESS = np.sum( (predicted_prices - np.mean(actual_prices)) ** 2)
        #Total sum of squares, equivalent to np.var() * len()
        #SST = np.var(actual_prices) * len(actual_prices)
        #SST = RSS + ESS
        SST = np.sum( (actual_prices - np.mean(actual_prices)) **2)
        #Explained variance score: 1 is perfect prediction
	try:
        	score = 1. - RSS / SST 
        	logging.info('Coefficient of determination {dataset} score:\t\t\t\t {score:>8}'.format(dataset=df,
                                                    score=score    ))
	except Exception as e:
		logging.info('Exception e: {}'.format(e))
        

    logging.info('Train Loss :{}'.format(cost_function(optimal_theta, train_x, train_y)))

if __name__ == '__main__':
    #dataset = load_data(input_file='housing.data')
    #np.random.shuffle(dataset) 
    #np.save('./dataset.npy',dataset)
    #dataset = np.load('./dataset.npy')
    dataset = np.load('./ex4_dataset.npy')
    
    
    orig_train_x, train_y, orig_test_x, test_y = split_data(dataset)
    
    standlizer = featureStandlizer(orig_train_x,'stand')

    train_x = standlizer.transform(orig_train_x)
    train_x = np.hstack([np.ones((train_x.shape[0], 1)), train_x])

    #orig_train_x = np.hstack([np.ones((orig_train_x.shape[0], 1)), orig_train_x])
    
    test_x = standlizer.transform(orig_test_x)
    test_x = np.hstack([np.ones((test_x.shape[0], 1)), test_x])

    #orig_test_x = np.hstack([np.ones((orig_test_x.shape[0], 1)), orig_test_x])
    
    theta_init = np.random.rand(train_x.shape[1])

    print 'train_x:', train_x, 'train_y:',train_y

    
    #hand write gradient descend
    optimal_theta_1 = gradient_descend(train_x, train_y, test_x, test_y, theta_init, sv_name='./results/ex4_gd_stand_normalized.png', iters=1000)
    optimal_theta_1 = newton_descend(train_x, train_y, test_x, test_y, theta_init, sv_name='./results/ex4_newton_stand_normalized.png', iters=2000)
    #optimal_theta_2 = scipy_optimize(train_x, train_y, test_x, test_y, theta_init, sv_name='./results/bfgs_max_normalized.png',iters=200)
    #optimal_theta_3 = normal_equation(train_x, train_y, test_x, test_y, sv_name='./results/normequ_stand_normalized.png')

    #optimal_theta_4 = gradient_descend(orig_train_x, train_y, orig_test_x, test_y, theta_init, sv_name='./results/gt_vs_pred_4.png', iters=200)
    #optimal_theta_5 = scipy_optimize(orig_train_x, train_y, orig_test_x, test_y, theta_init, sv_name='./results/gt_vs_pred_5.png',iters=200)
    #optimal_theta_6 = normal_equation(orig_train_x, train_y, orig_test_x, test_y, sv_name='./results/gt_vs_pred_6.png')

    #optimal_theta_7 = scipy_optimize(orig_train_x, train_y, orig_test_x, test_y, theta_init, sv_name='./results/gt_vs_pred_7.png',iters=200, _method='Powell')

    #optimal_theta_8 = linear_regression_sklearn(orig_train_x, train_y, orig_test_x, test_y, theta_init, sv_name='./results/sklearn_linear_gt_vs_pred_8.png')

    #optimal_theta_9 = linear_regression_sklearn(orig_train_x, train_y, orig_test_x, test_y, theta_init, sv_name='./results/sk_linear_regression_stand_normlized.png')
    #optimal_theta_9 = ridge_regression_sklearn(orig_train_x, train_y, orig_test_x, test_y, theta_init, sv_name='./results/sk_ridge_regression_stand_normlized.png')
    #optimal_theta_9 = ridge_regression_cv_sklearn(orig_train_x, train_y, orig_test_x, test_y, theta_init, sv_name='./results/sk_ridge_regression_cv_stand_normlized.png')
    
