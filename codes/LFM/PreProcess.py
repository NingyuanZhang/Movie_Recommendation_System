# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:25:44 2019

@author: Han Wu
"""

import numpy as np
import pandas as pd





if __name__ == "__main__":
	
	filepath2 = "train_df.csv"
	traindf = pd.read_csv(filepath2)
	filepath3 = "test_df.csv"
	testdf = pd.read_csv(filepath3)
	
	filepath = "./ml-latest-small/ratings.csv"
	df = pd.read_csv(filepath)
	num_users = len(df.userId.unique())
	num_movies = len(df.movieId.unique())
	indexM_RI = {}    # from movieId to Raw indexId, indexId starts from 1
	n = 0
	for i in df.itertuples():
		if i[2] not in indexM_RI:
			n += 1
			indexM_RI[i[2]] = n
	
	UtoM_train0 = np.zeros((num_users, num_movies))   # rows represent users;  columns represent movies
	UtoM_train_e0 = np.zeros((num_users, num_movies))
	Time_train0 = np.zeros((num_users, num_movies))
	for i in traindf.itertuples():
		UtoM_train0[i[1]-1, indexM_RI[i[2]]-1] = i[3]
		Time_train0[i[1]-1, indexM_RI[i[2]]-1] = i[4]
		UtoM_train_e0[i[1]-1, indexM_RI[i[2]]-1] = 1
		
	UtoM_test0 = np.zeros((num_users, num_movies))
	UtoM_test_e0 = np.zeros((num_users, num_movies))
	Time_test0 = np.zeros((num_users, num_movies))
	for i in testdf.itertuples():
		UtoM_test0[i[1]-1, indexM_RI[i[2]]-1] = i[3]
		Time_test0[i[1]-1, indexM_RI[i[2]]-1] = i[4]
		UtoM_test_e0[i[1]-1, indexM_RI[i[2]]-1] = 1
		
	numRperM = np.sum(UtoM_train_e0, axis=0, keepdims=True)
	quli = numRperM >= 4   # if one movie's ratings is >= 4, then this movie is qualified
	quli = quli + 0
	indexRI_I = {}    # from Raw_indexId-1 to indexId, indexId starts from 0
	n = 0
	for i in range(num_movies):
		if quli[0, i] > 0:
			if i not in indexRI_I:
				indexRI_I[i] = n
				n += 1
				
	R_train = np.zeros((num_users, n))
	E_train = np.zeros((num_users, n))
	Time_train = np.zeros((num_users, n))
	for i in range(num_movies):
		if quli[0][i] > 0:
			for x in range(num_users):
				if UtoM_train_e0[x, i] == 1:
					R_train[x, indexRI_I[i]] = UtoM_train0[x, i]
					E_train[x, indexRI_I[i]] = 1
					Time_train[x, indexRI_I[i]] = Time_train0[x, i]
					
	R_test = np.zeros((num_users, n))
	E_test = np.zeros((num_users, n))
	Time_test = np.zeros((num_users, n))
	for i in range(num_movies):
		if quli[0, i] > 0:
			for x in range(num_users):
				if UtoM_test_e0[x, i] == 1:
					R_test[x, indexRI_I[i]] = UtoM_test0[x, i]
					E_test[x, indexRI_I[i]] = 1
					Time_test[x, indexRI_I[i]] = Time_test0[x, i]
					
	np.savetxt("R_train.txt", R_train)
	np.savetxt("E_train.txt", E_train)
	np.savetxt("Time_train.txt", Time_train)
	
	np.savetxt("R_test.txt", R_test)
	np.savetxt("E_test.txt", E_test)
	np.savetxt("Time_test.txt", Time_test)
	
	moviT = {}
	for i in range(n):
		if i not in moviT:
			moviT[i] = []
		for x in range(num_users):
			if Time_train[x, i] > 0:
				moviT[i].append(Time_train[x, i])		
	moviT_medi = np.zeros((1,n))
	for i in range(n):
		moviT_medi[0, i] = np.mean(moviT[i])
		if i % 100 == 0:
			print(i)
	
	np.savetxt("moviT_medi.txt", moviT_medi)
	
	