import numpy as np
from scipy import stats as st
from scipy.stats import pearsonr

def gen_var_random(Nsamples, Nfeatures):
    """Generate a set of random variables and a random target variable.
    
    The target variable is thus uncorrelated with the "explanatory" variables.
    """
    X = st.norm().rvs(size=(Nsamples, Nfeatures))
    y = st.norm().rvs(size=Nsamples)
    return X, y

def gen_var_correlated(Nsamples, Nfeatures, noise=10):
    """Generate a set of variables with a correlated target variable.
    
    The target variable is generated from a linear relation with random
    coefficients plus a noise component with an amplitude of
    noise per cent the range of the noise-free variable.
    """
    X = st.norm().rvs(size=(Nsamples, Nfeatures))    
    fact = np.random.randint(-50, 50, Nfeatures)
    yc = X@fact
    scale = (yc.max() - yc.min())*noise/100
    yc = yc + np.random.normal(loc=0, scale=10, size=Nsamples)    
    return X, yc

def coef_det(y_true, y_pred):
    """Compute the coefficient of determination of the prediction.
    """
    u = ((y_true - y_pred)**2).sum()
    v = ((y_true - y_true.mean())**2).sum()
    c = 1 - u/v
    return c

def coef_correl(X, y):
    c = []
    for x in X.transpose():
        c.append(pearsonr(x, y))
    return np.array(c)