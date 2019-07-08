import pandas as pd
import numpy as np
import random
import math
import json

reverse_map = {}

def get_dataset(no_users):
    data = pd.read_csv('ratings.csv')
    # print(data.dtypes)
    R_train = np.zeros((no_users, 9742))
    R_test = np.zeros((no_users, 9742))

    movieId_dict = {}
    unique_count = 0
    for index, row in data.iterrows():
        userId = int(row.userId) - 1
        if userId == no_users:
            break
        movieId = int(row.movieId)
        rating = row.rating

        if movieId not in movieId_dict:
            movieId_dict[movieId] = unique_count
            reverse_map[unique_count] = movieId
            unique_count += 1

        col = movieId_dict[movieId]
        if random.uniform(0, 1) <= 0.8:
            R_train[userId, col] = rating
        else:
            R_test[userId, col] = rating

    print(R_train.shape)
    print(R_test.shape)
    return R_train, R_test
    # nonzero_r, nonzero_c = np.nonzero(R_train)
    # for x, y in zip(nonzero_r, nonzero_c):
    #     print(R_train[x, y])


def train_data(R_train, R_test, lr, ld1, ld2, no_factors=25, no_steps=50):
    # initialize p and q
    rows, cols = R_train.shape
    P = np.random.rand(rows, no_factors)
    Q = np.random.rand(no_factors, cols)


    print('test data before traing:')
    test_data(P, Q, R_test)

    # first find non_zeros
    nonzero_r, nonzero_c = np.nonzero(R_train)

    # traverse  all non_zeros
    steps = 0
    while steps < no_steps:
        steps += 1
        error = 0.0
        for row, col in zip(nonzero_r, nonzero_c):
            r = R_train[row, col]
            r_predicted = np.dot(P[row, :], Q[:, col])
            error += (r - r_predicted) ** 2
            # print('error', error)

        error = math.sqrt(error/ len(nonzero_r))
        print(steps, error)

        for row, col in zip(nonzero_r, nonzero_c):
            r = R_train[row, col]
            p_x = P[row, :]
            q_i = Q[:, col]

            ep = 2 * (r - np.dot(p_x, q_i))
            # print('ep', ep)
            # update
            # print('error', error)
            # print('before P', P[row, :])
            # print('before Q', Q[:, col])
            # print(P[row, :])
            P[row, :] = p_x + lr * (ep * q_i - ld1 * p_x)
            Q[:, col] = q_i + lr * (ep * p_x - ld2 * q_i)
            # print('after P', P[row, :])
            # print('after Q', Q[:, col])
            # break
        # print('after error', 2 * (r - np.dot(P[row, :], Q[:, col])))
    test_data(P, Q, R_test)
    return P, Q

def test_data(P, Q, R_test):
    # first find non_zeros
    nonzero_r, nonzero_c = np.nonzero(R_test)

    # traverse
    RMSE = 0.0
    MAE = 0.0
    for row, col in zip(nonzero_r, nonzero_c):
        r = R_test[row, col]
        r_predicted = np.dot(P[row, :], Q[:, col])
        RMSE += (r - r_predicted) ** 2
        MAE += abs(r - r_predicted)
    RMSE = math.sqrt(RMSE/ len(nonzero_r))
    MAE = MAE / len(nonzero_r)
    print('RMSE:', RMSE)
    print('MAE', MAE)


def get_top_ten(R_test, P, Q):
    nonzero_r, nonzero_c = np.nonzero(R_test)
    R_to_fill = R_test.copy()

    for row, col in zip(nonzero_r, nonzero_c):
        R_to_fill[row, col] = np.dot(P[row, :], Q[:, col])

    top_ten_lists = {}
    for i in range(len(R_to_fill)):
        user_row = R_to_fill[i]
        indices = np.argsort(user_row)[::-1]

        count = 0
        movies = []
        for index in indices:
            if index not in reverse_map:
                continue
            count += 1
            movies.append(index)
            if count == 10:
                break

        top_ten_lists[i] = movies


    # print(top_ten_lists[0])
    return top_ten_lists

def get_recommend_list(R_train, R_test, P, Q):
    R_to_fill = R_train.copy()
    nonzero_r, nonzero_c = np.nonzero(R_train)
    zero_r, zero_c = np.nonzero(R_train == 0)

    # fill missing data
    for row, col in zip(zero_r, zero_c):
        R_to_fill[row, col] = np.dot(P[row, :], Q[:, col])
    # print(R_to_fill[0, :20])
    # print(R_train[0, :20])


    recommend_dict = {}

    for i in range(len(R_to_fill)):
        user_row = R_to_fill[i]
        indices = np.argsort(user_row)[::-1]

        # get 10 top ranking movies
        count = 0
        ten_movies_list = []
        for index in indices:
            # if the item has been rated, skip it
            if R_train[i, index] != 0.0:
                continue

            # if the item is not rated, put in dict
            ten_movies_list.append(index)
            count += 1
            if count == 10:
                break

        recommend_dict[i] = ten_movies_list


    # print(recommend_dict[0])
    # exDict = {'exDict': exDict}

    return recommend_dict


def write_to_csv(recommend_dict):
    user_id = []
    movies_list = {}
    for i in range(10):
        movies_list[i] = []

    for key in recommend_dict.keys():
        # print(recommend_dict[key][i])
        user_id.append(key)
        for i in range(len(recommend_dict[key])):
            this_movie = recommend_dict[key][i]
            movie_id = reverse_map[this_movie]
            # print(movie_id)
            movies_list[i].append(movie_id)

    dict = {'user_id': user_id}
    for i in range(10):
        dict[('movies'+str(i + 1))] = movies_list[i]

    # print(dict)


    df = pd.DataFrame(dict)

    # saving the dataframe
    df.to_csv('file1.csv')




def cal_evaluation(R_train, R_test, P, Q):
    nonzero_r_test, nonzero_c_test = np.nonzero(R_test)
    nonzero_r_train, nonzero_c_train = np.nonzero(R_train)


    # prediction
    prediction_dict = {}
    R_to_fill = R_test.copy()
    for row, col in zip(nonzero_r_test, nonzero_c_test):
        prediction = np.dot(P[row, :], Q[:, col])
        R_to_fill[row, col] = prediction

    # print(R_to_fill[0, :10])
    # print(R_test[0, :10])

    # sort
    precision = 0.0
    recall = 0.0
    avg = 0.0
    DCG = 0.0
    IDCG = 0.0
    zero_rows = 0.0
    for i in range(len(R_to_fill)):
        if np.count_nonzero(R_test[i, :]) == 0.0:
            zero_rows += 1
            continue
        pred_row = R_to_fill[i]
        real_row = R_test[i]

        pred_indices = np.argsort(pred_row)[::-1]

        real_indices = np.argsort(real_row)[::-1]
        # print('same', len(set(pred_indices[:10]) & set(real_indices[:10])))
        # print(pred_indices[:10])
        # print(real_indices[:10])
        no_same_elements = len(set(pred_indices[:10]) & set(real_indices[:10]))

        # same_index = []
        # if no_same_elements > 0 :
        #     for x in pred_indices:
        #         for y in real_indices:
        #             if x == y:
        #                 same_index.append(x)
        both = set(pred_indices).intersection(real_indices)
        same_index = []
        for b in both:
            temp, = np.where(pred_indices == b)
            same_index.append(temp)


        DCG += sum([1 / math.log2(1 + p + 1) for p in same_index])
        IDCG += sum([1 / math.log2(1 + p + 1) for p in range(len(same_index))])


        precision += no_same_elements / 10.0
        recall += no_same_elements / np.count_nonzero(R_test[i, :])
        # print(precision, recall)
        # return

        # avg += np.count_nonzero(R_test[i, :])
        # print(precision)
        # return

    no_sample = len(R_to_fill) - zero_rows
    precision /= no_sample
    recall /= no_sample
    DCG /= no_sample
    IDCG /= no_sample
    print('precision:', precision)
    print('recall:', recall)
    print('F-measures:', recall * precision / (recall + precision))
    print('NDCG', DCG / IDCG)


if __name__ == '__main__':
    R_train, R_test = get_dataset(100)
    P,Q = train_data(R_train, R_test, 0.01, 0.2, 0.2)
    cal_evaluation(R_train, R_test, P, Q)
    # recommend_dict = get_recommend_list(R_train, R_test, P, Q)
    recommend_dict = get_top_ten(R_test, P, Q)
    write_to_csv(recommend_dict)
    # cal_evaluation(recommend_dict, R_test)
