# -*- coding: utf-8 -*-
"""
Created on Sat May 11 21:41:29 2019

@author: Han Wu
"""

import numpy as np

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


def predictr(P, Q):
	pred = np.dot(P,Q)
	return pred


def recommendation(pred, etrain):
	u,m = etrain.shape
	rec = np.zeros((u,10))
	for x in range(u):
		mi = np.argsort(pred[x])
		mi = np.flip(mi, axis=0)
		rec[x] = mi[:10]
	return rec



def calcprecision(Rte, Ete, rec):
	u,m = Rte.shape
	precM = np.zeros((u,10))
	for x in range(u):
		for i in range(10):
			if Ete[x, int(rec[x,i])] == 1:
				precM[x, i] = 1
	precu = np.sum(precM, axis=1) / 10
	prec = np.mean(precu)
	return precM, prec


def calcrecall(precM, Etest):
	u,m = Etest.shape
	precx = np.sum(precM, axis=1)
	Ex = np.sum(Etest, axis=1)
	s = 0
	for x in range(u):
		if Ex[x] != 0:
			s += precx[x] / Ex[x]
	recall = s / u
	return recall



def calcCG(precM, Etest):
	u,m = Etest.shape
	ndcgL = []
	ndcg_index = []
	precx = np.sum(precM, axis=1)
	for x in range(u):
		if precx[x] != 0:
			ndcg_index.append(x)
			dcg = calcDcg(precM[x])
			idcg = calcIdcg(precM[x])
			ndcg = dcg / idcg
			ndcgL.append(ndcg)
	return ndcgL, ndcg_index


def calcDcg(L):
	s = 0
	for i in range(10):
		s += L[i] / np.log2(i+2)
	return s

def calcIdcg(L):
	l1 = np.sort(L)
	l1 = np.flip(l1, axis=0)
	return calcDcg(l1)




if __name__ == "__main__":
	
	Rtrain = np.loadtxt("R_train0.txt")
	Etrain = np.loadtxt("E_train0.txt")
	
	Rtest = np.loadtxt("R_test0.txt")
	Etest = np.loadtxt("E_test0.txt")
	
	P = np.loadtxt("p0.txt")
	Q = np.loadtxt("q0.txt")
	print('Load data complete.')

	maef = compMAE(Rtest, P, Q, Etest)
	print("The mean absolute error is:", maef)
	
	rmsef = compRMSE(Rtest, P, Q, Etest)
	print("The root mean square error is:", rmsef)
	
	print("Load data complete.")
	
	predr = predictr(P, Q)
	
	recomm = recommendation(predr, Etrain)
	print('Recommendation complete.')
	
	precM, precision = calcprecision(Rtest, Etest, recomm)
	print('Precision complete.')
	
	recall = calcrecall(precM, Etest)
	print('Recall complete.')
	
	F = 2 * precision * recall / (precision + recall)
	print('F complete.')
	
	
	ndcglist, ndcg_index = calcCG(precM, Etest)
	ndcgA = np.mean(ndcglist)
	print('NDCG complete.')