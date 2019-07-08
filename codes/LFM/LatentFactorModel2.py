# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:06:03 2019

@author: Han Wu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def initialization(u, m, num_factors, nb):
	P = np.random.randn(u, num_factors)
	Q = np.random.randn(num_factors, m)
	Bu = np.random.randn(u, 1)
	Bm = np.random.randn(nb, m)
	
	return P, Q, Bu, Bm


def compRMSE(R, P, Q, E, mT, mTavg, Bu, Bm, mu):
	N = np.sum(E)  # number of examples
	u,m = R.shape
	Bt = np.zeros((u,m))
	for x in range(u):
		for i in range(m):
			if mT[x, i] < mTavg[0, i]:
				Bt[x, i] = Bm[0, i]
			else:
				Bt[x, i] = Bm[1, i]
	
	predict = ( np.dot(P,Q) + mu + Bu + Bt ) * E
	M = (R - predict) * E
	cost = 1/N * np.power(np.linalg.norm(M), 2)
	rmse = np.power(cost, 0.5)
	return rmse


def compGrad(R, P, Q, E, mT, mTavg, x, i, Bu, Bm, lamda1, lamda2, lamda3, lamda4):
	N = np.sum(E)  # number of training examples
	r_total = np.sum(R)
	mu = r_total / N
	
	t=0
	if mT[x, i] < mTavg[0, i]:
		bt = Bm[0, i]
	else:
		bt = Bm[1, i]
		t = 1
	
	epsilon = 2 * (R[x][i] - mu - Bu[x][0] - bt - np.dot(P[[x],:], Q[:,[i]]))
	epsilon = epsilon.squeeze()
	gradPx = -1 * epsilon * Q[:,[i]].T + 2 * lamda1 * P[[x],:]
	gradQi = -1 * epsilon * P[[x],:].T + 2 * lamda2 * Q[:,[i]]
	gradBux = -1 * epsilon + 2 * lamda3 * Bu[x][0]
	gradBmti = -1 * epsilon + 2 * lamda4 * bt
	
	return gradPx, gradQi, gradBux, gradBmti, t



def LFModel2(R, E, mT, mTavg, lamda1, lamda2, lamda3, lamda4, learning_rate, num_iter, num_factors, R_test, E_test, mT_test):
	u, m = R.shape
	nbin = 2
	P, Q, Bu, Bm = initialization(u, m, num_factors, nbin)
	rall = np.sum(R)
	N = np.sum(E)
	mu = rall / N     # the overall average rating
	rmseA = []
	rmseA_test = []
	iter_show = []
	n = 0
	for j in range(num_iter):
		
		for x in range(u):
			
			for i in range(m):
				if E[x][i] != 0:
					'''
					gradPx, gradQi, gradBux, gradBmti, t = compGrad(R, P, Q, E, mT, mTavg, x, i, Bu, Bm, lamda1, lamda2, lamda3, lamda4)
					P[[x],:] = P[[x],:] - learning_rate * gradPx
					Q[:,[i]] = Q[:,[i]] - learning_rate * gradQi
					Bu[x][0] = Bu[x][0] - learning_rate * gradBux
					Bm[t, i] = Bm[t, i] - learning_rate * gradBmti
					
					rmse = compRMSE(R, P, Q, E, mT, mTavg, Bu, Bm, mu)
					rmse_test = compRMSE(R_test, P, Q, E_test, mT_test, mTavg, Bu, Bm, mu)
					rmseA.append(rmse)
					rmseA_test.append(rmse_test)
					iter_show.append(n)
					print("n =", n)
					print("rmse =", rmse)
					print("rmseTest =", rmse_test)
					print()
					n+= 1
					'''
					
					if n % 100 == 0:
						rmse = compRMSE(R, P, Q, E, mT, mTavg, Bu, Bm, mu)
						rmse_test = compRMSE(R_test, P, Q, E_test, mT_test, mTavg, Bu, Bm, mu)
						rmseA.append(rmse)
						rmseA_test.append(rmse_test)
						iter_show.append(n)
						print("n =", n)
						print("rmse =", rmse)
						print("rmseTest =", rmse_test)
						print()
					
					gradPx, gradQi, gradBux, gradBmti, t = compGrad(R, P, Q, E, mT, mTavg, x, i, Bu, Bm, lamda1, lamda2, lamda3, lamda4)
					P[[x],:] = P[[x],:] - learning_rate * gradPx
					Q[:,[i]] = Q[:,[i]] - learning_rate * gradQi
					Bu[x][0] = Bu[x][0] - learning_rate * gradBux
					Bm[t, i] = Bm[t, i] - learning_rate * gradBmti
					n += 1
					
		print("j =", j)
		print()
	return P, Q, Bu, Bm, rmseA, rmseA_test, iter_show

if __name__ == "__main__":
	
	Rtrain = np.loadtxt("R_train.txt")
	Etrain = np.loadtxt("E_train.txt")
	Timetrain = np.loadtxt("Time_train.txt")
	
	Rtest = np.loadtxt("R_test.txt")
	Etest = np.loadtxt("E_test.txt")
	Timetest = np.loadtxt("Time_test.txt")
	
	numU, numM = Rtrain.shape
	
	moviTavg = np.loadtxt("moviT_medi.txt")	
	moviTavg = moviTavg.reshape((1,numM))
	
	lamda1 = 16
	lamda2 = 16
	lamda3 = 16
	lamda4 = 16
	learning_rate = 0.0002
	num_iter = 10
	num_factors = 250
	
	Pf, Qf, Buf, Bmf, rmseA, rmseA_test, iter_show = LFModel2(Rtrain, Etrain, Timetrain, moviTavg, lamda1, lamda2, lamda3, lamda4, learning_rate, num_iter, num_factors, Rtest, Etest, Timetest)
	rt = np.sum(Rtrain)
	N = np.sum(Etrain)
	mu = rt / N
	rmsef = compRMSE(Rtest, Pf, Qf, Etest, Timetest, moviTavg, Buf, Bmf, mu)
	print("The test root mean square error is:", rmsef)
	
	
	
	plt.figure()
	plt.plot(iter_show, rmseA, 'b', label='training error')
	plt.plot(iter_show, rmseA_test, 'r', label='test error')
	plt.xlabel('number of iterations')
	plt.ylabel('cost')
	plt.title('training error and test error vs num_iterations (temporal bias)')
	plt.legend()
	plt.savefig('./fig7.jpg')
	plt.show()
	
	np.savetxt("p27.txt", Pf)
	np.savetxt("q27.txt", Qf)
	np.savetxt("bu27.txt", Buf)
	np.savetxt("bm27.txt", Bmf)
	
	