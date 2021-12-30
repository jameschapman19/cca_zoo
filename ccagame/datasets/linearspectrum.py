
import numpy as np
from cca_zoo.data import generate_covariance_data
from sklearn.model_selection import train_test_split

def linear_dataset():
    X,Y=generate_covariance_data(1000,[50,50],latent_dims=50, correlation=np.linspace(0,1,50))
    X,X_te,Y,Y_te=train_test_split(X,Y,test_size=0.2)
    return X,X_te,Y,Y_te

