# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:47:27 2019

@author: Han Wu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def initialization(u, m, num_factors = 20):
	P = np.random.randn(u, num_factors)
	Q = np.random.randn(num_factors, m)
	
	return P, Q

def compRMSE(R, P, Q, E):
	N = np.sum(E)  # number of training examples
	M = (R - np.dot(P,Q)) * E
	cost = 1/N * np.power(np.linalg.norm(M), 2)
	rmse = np.power(cost, 0.5)
	return rmse

def compMAE(R, P, Q, E):
	N = np.sum(E)
	M = (R - np.dot(P,Q)) * E
	mae = 1/N * np.sum(abs(M))
	return mae

def compGrad(R, P, Q, E, lamda1, lamda2):
	N = np.sum(E)  # number of training examples
	M = (R - np.dot(P,Q)) * E
	gradP = -2/N * np.dot(M, Q.T) + 2/N * lamda1 * P
	gradQ = -2/N * np.dot(P.T, M) + 2/N * lamda2 * Q
	
	return gradP, gradQ

def LFModel0(R, E, lamda1, lamda2, learning_rate, num_iter, num_factors, R_test, E_test):
	u, m = R.shape
	P, Q = initialization(u, m, num_factors)
	rmseA = []
	rmseA_test = []
	for i in range(num_iter):
		
		gradP, gradQ = compGrad(R, P, Q, E, lamda1, lamda2)
		P = P - learning_rate * gradP
		Q = Q - learning_rate * gradQ
	
		if i % 100 == 0:
			rmse = compRMSE(R, P, Q, E)
			rmse_test = compRMSE(R_test, P, Q, E_test)
			rmseA.append(rmse)
			rmseA_test.append(rmse_test)
			print(i)
			print('rmse:', rmse)
			print('rmse_test:', rmse_test)
			print()
	
	return P, Q, rmseA, rmseA_test
	

if __name__ == "__main__":
	
	filepath = "./ml-latest-small/ratings.csv"
	df = pd.read_csv(filepath)
	num_users = len(df.userId.unique())
	num_movies = len(df.movieId.unique())
	
	#max_u = max(df.userId)
	#max_m = max(df.movieId)

	indexMI = {}    # from movieId to indexId, indexId starts from 1
	n = 0
	for i in df.itertuples():
		if i[2] not in indexMI:
			n += 1
			indexMI[i[2]] = n
	#print( [i for i in indexMI.keys()][-5:] )
	#print( [i for i in indexMI.values()][-5:] )

	from sklearn.model_selection import train_test_split
	train_df, test_df = train_test_split(df, test_size=0.2)
	
	UtoM_train = np.zeros((num_users, num_movies))   # rows represent users;  columns represent movies
	UtoM_train_e = np.zeros((num_users, num_movies))
	for i in train_df.itertuples():
		UtoM_train[i[1]-1, indexMI[i[2]]-1] = i[3]
		UtoM_train_e[i[1]-1, indexMI[i[2]]-1] = 1
		
	UtoM_test = np.zeros((num_users, num_movies))
	UtoM_test_e = np.zeros((num_users, num_movies))
	for i in test_df.itertuples():
		UtoM_test[i[1]-1, indexMI[i[2]]-1] = i[3]
		UtoM_test_e[i[1]-1, indexMI[i[2]]-1] = 1
	
	lamda1 = 20
	lamda2 = 20
	learning_rate = 0.1
	num_iter = 50001
	num_factors = 100
	
	Pf, Qf, rmseA, rmseA_test = LFModel0(UtoM_train, UtoM_train_e, lamda1, lamda2, learning_rate, num_iter, num_factors, UtoM_test, UtoM_test_e)
	np.savetxt("p0.txt", Pf)
	np.savetxt("q0.txt", Qf)
	
	np.savetxt("R_train0.txt", UtoM_train)
	np.savetxt("E_train0.txt", UtoM_train_e)
	
	np.savetxt("R_test0.txt", UtoM_test)
	np.savetxt("E_test0.txt", UtoM_test_e)
	
	maef = compMAE(UtoM_test, Pf, Qf, UtoM_test_e)
	print("The mean absolute error is:", maef)
	rmsef = compRMSE(UtoM_test, Pf, Qf, UtoM_test_e)
	print("The root mean square error is:", rmsef)
	print("lambda1 is:", lamda1)
	print("lambda2 is:", lamda2)
	print("Number of factors is:", num_factors)
	
	iterA = range(0, num_iter, 100)
	plt.figure()
	plt.plot(iterA, rmseA, 'b', label='training error')
	plt.plot(iterA, rmseA_test, 'r', label='test error')
	plt.xlabel('number of iterations')
	plt.ylabel('cost')
	plt.title('training error and test error vs num_iterations')
	plt.legend()
	plt.savefig('./fig00.jpg')
	plt.show()
	
	




