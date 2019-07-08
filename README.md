
# Movie_Recommendation_System
It's a movie recommendation ststem on MovieLens dataset.
## Overview
Our goal is to create a recommendation list for each user. This project consists three steps: 1) Data Processing, create the training and testing dataset from the original dataset; 2) Conduct rating prediction and make evaluation based on MAE and RMSE; 3) Conduct Top-N recommendation and make evaluation based on Precision, Recall, F-measure and NDCG.
## Dataset
The	dataset	can	be	downloaded	from	here:	
https://grouplens.org/datasets/movielens/latest/
It includes the	user_ID, item_ID, user-item	ratings, time, user-movie	tags,	as well as the metadata	of	the	movies,	such	as title	and	genre.	
## Data selection and preprocessing
For each user, randomly select 80% of his/her ratings as the training ratings, and use the remaining 20% ratings as testing ratings. The training ratings from all users consist the final training dataset, and the testing ratings from all users consist the final testing dataset.
## Rating prediction
Based on the training dataset, we use different models to conduct prediction.
After predicting the ratings in the test dataset( as if we didn't know them), we evaluate predictions by calculating the MAE and RMSE.
### User-User Collaborative Filtering
The mian idea is to  
<img src="http://chart.googleapis.com/chart?cht=tx&chl= 在此插入Latex公式" style="border:none;">
<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" style="border:none;">
### Item-Item Collaborative Filtering
### Basic Latent Factor Model
### Latent Factor Model with Biases
### Latent Factor Model with Temporal Biases
