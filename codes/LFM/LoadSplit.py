# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:36:55 2019

@author: Han Wu
"""

import numpy as np
import pandas as pd



if __name__ == "__main__":
	
	filepath = "./ml-latest-small/ratings.csv"
	df = pd.read_csv(filepath)
	num_users = len(df.userId.unique())
	num_movies = len(df.movieId.unique())

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
	
	train_df.to_csv("train_df.csv", index=False)
	test_df.to_csv("test_df.csv", index=False)



