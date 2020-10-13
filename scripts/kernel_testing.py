from CCA_methods.generate_data import generate_candola
from CCA_methods.linear import Wrapper
import matplotlib.pyplot as plt

X, Y = generate_candola(1000, 5, 100, 100, 0.5, 0.5)
#X = np.random.rand(1000, 100)
#X -= X.mean(axis=0)
#Y = np.random.rand(1000, 100)
#Y -= Y.mean(axis=0)

X_t, Y_t = X[700:], Y[700:]
X, Y = X[:700], Y[:700]

params = {'kernel': 'linear', 'reg': 0.00001}
a = Wrapper(2, method='kernel', params=params).fit(X, Y)
a_ = a.predict_corr(X_t, Y_t)
b = Wrapper(2, method='l2').fit(X, Y)
b_ = b.predict_corr(X_t, Y_t)
params = {'kernel': 'poly', 'degree': 2, 'reg': 0.00001}
c = Wrapper(2, method='kernel', params=params).fit(X, Y)
c_ = c.predict_corr(X_t, Y_t)
params = {'kernel': 'gaussian', 'gausigma': 2, 'reg': 0.00001}
d = Wrapper(2, method='kernel', params=params).fit(X, Y)
d_ = d.predict_corr(X_t, Y_t)
plt.show()
print('hello')