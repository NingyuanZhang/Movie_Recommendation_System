import numpy as np
import pandas as pd
import math
from collections import Counter

def constructlists(filename):
    print("construct lists...")
    df = pd.read_csv(filename)
    user_list = []
    movie_list = []
    for index, row in df.iterrows():
        #print(index, row['userId'],row['movieId'])
        if(row['userId'] not in user_list):
            user_list.append(row['userId'])
        if (row['movieId'] not in movie_list):
            movie_list.append(row['movieId'])
    user_list = sorted(user_list)
    movie_list = sorted(movie_list)
    return user_list, movie_list

def constructMatrix(filename, rows, cols,movie_list,user_list):
    print("construct rating matrix...")
    rating_matrix = np.zeros((rows,cols));
    for i in range(rows):
        for j in range (cols):
            rating_matrix[i][j] = -1;
    df = pd.read_csv(filename)
    for index, row in df.iterrows():
        user_index = user_list.index(row['userId'])
        movie_index = movie_list.index(row['movieId'])
        rating_matrix[movie_index][user_index] = row['rating']
    return rating_matrix

def constructAvgMatrix(rating_matrix,rows,cols):
    # compute avg for each ros
    print("compute avgs for each rows...")
    avgs = [0] * rows
    for i in range(rows):
        sum=0.0
        sum_count=0.0
        for j in range (cols):
            if (rating_matrix[i][j]>=0):
                sum+=rating_matrix[i][j]
                sum_count=sum_count+1
        if(sum_count!=0):
            avgs[i] = sum/sum_count

    #compute rating_matrix_avged
    print("compute rating_matrix_avged...")
    rating_matrix_avged = np.zeros((rows, cols));
    for i in range(rows):
        for j in range (cols):
            if (rating_matrix[i][j] >= 0):
                rating_matrix_avged[i][j]=rating_matrix[i][j]-avgs[i]
            else:
                rating_matrix_avged[i][j] = 0
    return rating_matrix_avged

def constructTrain(rating_matrix, rows, cols, ratio):
    train = np.zeros((rows,cols))
    for i in range(rows):
        for j in range (cols):
            if(i>=round(rows*ratio) and j>=round(cols*ratio)):
                train[i][j] = -1;
            else:
                train[i][j] = rating_matrix[i][j];
    return train

def predict(index_i,index_j,rating_matrix,rating_matrix_avged):
    print("predict", index_i,index_j,"...")
    #find the most similar items(2)
    sim_list = []
    for i in range(rows):
        if(rating_matrix[i][index_j]>=0 or i==index_i):
            multi = 0.0
            x = 0.0
            y = 0.0
            for j in range (cols):
                multi += rating_matrix_avged[i][j]*rating_matrix_avged[index_i][j]
                x += rating_matrix_avged[i][j]*rating_matrix_avged[i][j]
                y += rating_matrix_avged[index_i][j]*rating_matrix_avged[index_i][j]
            xy=math.sqrt(x)*math.sqrt(y)
            if(xy!=0):
                sim = multi/xy
            else:
                sim = 0.0
            if (i==index_i):
                sim_list.append(1.0)
            else:
                sim_list.append(sim)
        else:
            sim_list.append(0)

    idx = sorted(range(len(sim_list)), key=lambda k: sim_list[k])
    if(index_i==idx[len(idx)-1]):
        idx1 = idx[len(idx)-2]
        idx2 = idx[len(idx)-3]
    else:
        print("never reach here!")
        return 2.5

    sim1 = sim_list[idx1]
    sim2 = sim_list[idx2]
    val1 = rating_matrix[idx1][index_j]
    val2 = rating_matrix[idx2][index_j]
    if(sim1+sim2==0):
        #print("Not enough info to predict. Use neutrol rating.")
        predict = 2.5
    else:
        predict = (sim1*val1+sim2*val2)/(sim1+sim2)
    return predict

def evaluate1(rating_matrix,train, train_avged,rows,cols,ratio):
    for i in range(rows):
        for j in range (cols):
            if (i >= round(rows * ratio) and j >= round(cols * ratio)):
                if(rating_matrix[i][j]>=0):
                    train[i][j] = predict(i,j,train,train_avged)

    #print("predict",train)
    err1 = 0.0
    err2 = 0.0
    count = 0.0
    for i in range(rows):
        for j in range (cols):
            if (i >= round(rows * ratio) and j >= round(cols * ratio)):
                if(train[i][j]>=0):
                    diff = train[i][j] - rating_matrix[i][j]
                    #MAE
                    err1+= abs(diff)
                    #RMSE
                    err2+= diff*diff
                    count=count+1
    if(count!=0):
        MAE = err1/count
        RMSE = math.sqrt(err2/count)
        print("MAE",MAE)
        print("RMSE", RMSE)
    else:
        print("Nothing needs to predict. Please adjust ratio!")
    return train

def evaluate2(rating_matrix,train, train_avged,rows,cols,ratio):
    allRecommendations = 10
    for i in range(rows):
        print(i)
        for j in range (cols):
            if (j >= round(cols * ratio) and train[i][j]<0):
                    train[i][j] = predict(i,j,train,train_avged)

    #pd.DataFrame(train).to_csv("train_predicted.csv")
    precision_sum = 0.0
    recall_sum =0.0
    NDCG_sum = 0.0
    count = 0.0
    for j in range(cols):
        if (j >= round(cols * ratio) and j < round(cols * ratio+10)):
            hit = 0.0
            DCG = 0.0
            IDCG = 0.0
            goodmovies = 0.0
            if (j >= round(cols * ratio)):
                #find good and recommended movies and store them to recommend_list
                recommend_list = {}
                for i in range(rows):
                    if (rating_matrix[i][j] < 0 or i >= round(rows * ratio)):
                        recommend_list[i]=train[i][j]
                #get top 10 from recommend_list
                k = Counter(recommend_list)
                recommend_list = dict(k.most_common(allRecommendations))

                for i in range(rows):
                    if (i >= round(rows * ratio) and rating_matrix[i][j] >= 0):
                        if i in recommend_list:
                            hit=hit+1
                            DCG+=1/(math.log10(2+i)/math.log10(2))

                for i in range(int(hit)):
                    IDCG+=1/(math.log10(2+i)/math.log10(2))

                if(IDCG==0):
                    NDCG = 0
                else:
                    NDCG = DCG/IDCG
                NDCG_sum+=NDCG

                avg_rate = 0.0
                avg_count = 0.0
                for i in range(rows):
                    if(rating_matrix[i][j] >= 0):
                        avg_rate+=rating_matrix[i][j]
                        avg_count+=1
                avg_rate = avg_rate/avg_count
                for i in range(rows):
                    if (rating_matrix[i][j] >= avg_rate):
                        goodmovies+=1

                precision_sum+=hit/allRecommendations
                recall_sum+=hit/goodmovies
                count+=1

    if(count!=0):
        precision = precision_sum/count
        recall = recall_sum/count
        NDCG = NDCG_sum/count
    else:
        precision = 0
        recall = 0
        NDCG = 0
    if((precision + recall)!=0):
        F = 2 * precision * recall / (precision + recall)
    else:
        F= 0
    print("precision",precision)
    print("recall", recall)
    print("F", F)
    print("NDCG", NDCG)

user_list, movie_list = constructlists(r'ratings1.csv')
rows = len(movie_list)
cols = len(user_list)
rating_matrix = constructMatrix(r'ratings1.csv', rows, cols, movie_list,user_list)
ratio = 0.9
train = constructTrain(rating_matrix, rows, cols, ratio)
#pd.DataFrame(train).to_csv("train.csv")
train_avged = constructAvgMatrix(train,rows,cols)
#pd.DataFrame(train_avged).to_csv("train_avged.csv")
predict_matrix = evaluate1(rating_matrix,train, train_avged,rows,cols,ratio)
evaluate2(rating_matrix,train, train_avged,rows,cols,ratio)





