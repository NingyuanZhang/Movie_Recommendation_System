import numpy as np
import csv
import os
from os import path

CWD = os.getcwd()
_DIR_DATASET = path.join(CWD, 'dataset')
_DIR_FULL = path.join(_DIR_DATASET, 'full')
_DIR_SMALL = path.join(_DIR_DATASET, 'small')

_FILE_RATINGS_SMALL = path.join(_DIR_SMALL, 'ml-latest-small/ratings.csv')
_FILE_RATINGS_FULL = path.join(_DIR_FULL, 'ml-latest/ratings.csv')

def generate_matrix(fp, if_save=False):
    f = open(fp)
    data = csv.reader(f)
    # skip header
    next(data)

    userIDs = set()
    movieIDs = set()
    ratings = []

    for row in data:
        user_id, movie_id, rating, _ = row
        userIDs.add(int(user_id))
        movieIDs.add(int(movie_id))
        ratings.append((int(user_id), int(movie_id), float(rating)))
    
    userIDs = list(userIDs)
    movieIDs = list(movieIDs)
    userIDs.sort()
    movieIDs.sort()
    
    row = len(userIDs)
    col = len(movieIDs)

    R = np.empty((row, col))
    R[:] = np.nan

    count = 0
    total = len(ratings)

    for rating in ratings:
        count += 1
        print("Progress: %d / %d" % (count, total), end='\r')
        uid, mid, ra = rating
        r = userIDs.index(uid)
        c = movieIDs.index(mid)
        R[r, c] = ra
    
    print('\nProcess complete.')
    
    if if_save:
        fo = path.join(_DIR_SMALL, 'R')
        np.save(fo, R)

generate_matrix(_FILE_RATINGS_SMALL, True)
generate_matrix(_FILE_RATINGS_FULL, True)