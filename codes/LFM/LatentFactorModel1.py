# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:26:10 2019

@author: Han Wu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def initialization(u, m, num_factors = 20):
	P = np.random.randn(u, num_factors)
	Q = np.random.randn(num_factors, m)
	Bu = np.random.randn(u, 1)
	Bm = np.random.randn(m, 1)
	
	return P, Q, Bu, Bm


def compGrad(R, P, Q, E, x, i, Bu, Bm, lamda1, lamda2, lamda3, lamda4):
	N = np.sum(E)  # number of training examples
	r_total = np.sum(R)
	mu = r_total / N
	
	epsilon = 2 * (R[x][i] - mu - Bu[x][0] - Bm[i][0] - np.dot(P[[x],:], Q[:,[i]]))
	epsilon = epsilon.squeeze()
	gradPx = -1 * epsilon * Q[:,[i]].T + 2 * lamda1 * P[[x],:]
	gradQi = -1 * epsilon * P[[x],:].T + 2 * lamda2 * Q[:,[i]]
	gradBux = -1 * epsilon + 2 * lamda3 * Bu[x][0]
	gradBmi = -1 * epsilon + 2 * lamda4 * Bm[i][0]
	
	return gradPx, gradQi, gradBux, gradBmi


def compRMSE(R, P, Q, E, Bu, Bm, mu):
	N = np.sum(E)  # number of training examples
	M = (R - mu - Bu - Bm.T - np.dot(P,Q)) * E
	cost = 1/N * np.power(np.linalg.norm(M), 2)
	rmse = np.power(cost, 0.5)
	return rmse


def LFModel1(R, E, lamda1, lamda2, lamda3, lamda4, learning_rate, num_iter, num_factors, R_test, E_test):
	u, m = R.shape
	P, Q, Bu, Bm = initialization(u, m, num_factors)
	r_total = np.sum(R)
	N = np.sum(E)
	mu = r_total / N     # the overall average rating
	#P = np.loadtxt("p.txt")
	#Q = np.loadtxt("q.txt")
	rmseA = []
	rmseA_test = []
	iter_show = []
	n = 0
	for j in range(num_iter):
		for x in range(u):
			for i in range(m):
				if E[x][i] != 0:
					if n % 100 == 0:
						rmse = compRMSE(R, P, Q, E, Bu, Bm, mu)
						rmse_test = compRMSE(R_test, P, Q, E_test, Bu, Bm, mu)
						rmseA.append(rmse)
						rmseA_test.append(rmse_test)
						iter_show.append(n)
						print("n =", n)
						print("rmse =", rmse)
						print("rmseTest =", rmse_test)
						print()
					
					gradPx, gradQi, gradBux, gradBmi = compGrad(R, P, Q, E, x, i, Bu, Bm, lamda1, lamda2, lamda3, lamda4)
					P[[x],:] = P[[x],:] - learning_rate * gradPx
					Q[:,[i]] = Q[:,[i]] - learning_rate * gradQi
					Bu[x][0] = Bu[x][0] - learning_rate * gradBux
					Bm[i][0] = Bm[i][0] - learning_rate * gradBmi
					n += 1
		print("j =", j)
		print()
	return P, Q, Bu, Bm, rmseA, rmseA_test, iter_show


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
	
	lamda1 = 0.01
	lamda2 = 0.01
	lamda3 = 0.1
	lamda4 = 0.1
	learning_rate = 0.003
	num_iter = 4
	num_factors = 150
	
	
	Pf, Qf, Buf, Bmf, rmseA, rmseA_test, iter_show = LFModel1(UtoM_train, UtoM_train_e, lamda1, lamda2, lamda3, lamda4, learning_rate, num_iter, num_factors, UtoM_test, UtoM_test_e)
	np.savetxt("p1.txt", Pf)
	np.savetxt("q1.txt", Qf)
	#maef = compMAE(UtoM_test, Pf, Qf, UtoM_test_e)
	#print("The mean absolute error is:", maef)
	rtrain_total = np.sum(UtoM_train)
	N = np.sum(UtoM_train_e)
	mu = rtrain_total / N
	rmsef = compRMSE(UtoM_test, Pf, Qf, UtoM_test_e, Buf, Bmf, mu)
	print("The root mean square error is:", rmsef)
	print("lambda1 is:", lamda1)
	print("lambda2 is:", lamda2)
	print("lambda2 is:", lamda3)
	print("lambda2 is:", lamda4)
	print("Number of factors is:", num_factors)
	
	plt.figure()
	plt.plot(iter_show, rmseA, 'b', label='training error')
	plt.plot(iter_show, rmseA_test, 'r', label='test error')
	plt.xlabel('number of iterations')
	plt.ylabel('cost')
	plt.title('training error and test error vs num_iterations (with bias)')
	plt.legend()
	plt.savefig('./fig4.jpg')
	plt.show()
	
	