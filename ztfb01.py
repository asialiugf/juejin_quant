import numpy as np
np.random.seed(1000)
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt

def gen_path(S0,r,sigma,T,M,I):
    dt = float(T)/M
    paths = np.zeros((M+1,I),np.float64)
    paths[0] = S0
    for t in range(1,M+1):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean())/rand.std()
        paths[t] = paths[t-1]*np.exp((r - 0.5*sigma**2)*dt + sigma * np.sqrt(dt) * rand)
    return paths

S0 = 100
r = 0.05
sigma = 0.2
T = 1.0
M = 50
I = 250000
paths = gen_path(S0,r,sigma,T,M,I)
plt.plot(paths[:,:10])
plt.show()

log_returns = np.log(paths[1:]/paths[0:-1])#from second to the end dividend by the first to the last second one to get the return rate

def print_statistics(array):
    sta = scs.describe(array)
    #print sta[0]
    print ("%14s %15.5f" % ('size',sta[0]))
    print ("%14s %15.5f" % ('min',sta[1][0]))
    print ("%14s %15.5f" % ('max',sta[1][1]))
    print ("%14s %15.5f" % ('mean',sta[2]))
    print ("%14s %15.5f" % ('std',np.sqrt(sta[3])))
    print ("%14s %15.5f" % ('skew',sta[4]))
    print ("%14s %15.5f" % ('kurtosis',sta[5]))

print_statistics(log_returns.flatten())

plt.hist(log_returns.flatten(),bins = 70,normed=True)
x = np.linspace(plt.axis()[0],plt.axis()[1])
plt.plot(x, scs.norm.pdf(x,loc = r/M,scale = sigma/np.sqrt(M)),'r',lw = 2)
plt.show()
sm.qqplot(log_returns.flatten()[::500],line='s')
plt.show()