import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from astropy.io.fits import getdata
from pln_distrib import pln

def objfunc(x, p):
    if p[2] <= 0:
        return np.inf
    vals = -np.sum(pln.logpdf(x, *p))
    return vals

# img = getdata("../testingdata/chamaeleonI-250.fits")
# sampledata = img[np.isfinite(img)]
# sampledata = sampledata + np.abs(sampledata.min()) + 0.1 # [sampledata > 0]


# sampledata = np.exp(np.random.randn(1000) + 2)
sampledata = pln.rvs(2, 2, 0.1, size=1000) + 0.1 * np.random.randn(1000)

guess = [1.8, 2.1, 0.3]

params = opt.minimize(
    lambda p: objfunc(sampledata, p), guess, method='Powell')
print("Finished minimization.")
p = params['x']
plt.hist(sampledata, bins=100, normed=1, log=True)
trialx = np.linspace(sampledata.min(), sampledata.max(), 1000)
plt.plot(trialx, pln.logpdf(trialx, *p), "b-", label="Minimize Powell")

params2 = opt.basinhopping(lambda p: objfunc(sampledata, p), guess,
                           minimizer_kwargs={"method": "Powell"}, niter=100)
print("Finished basinhopping.")
p2 = params2['x']
plt.plot(trialx, pln.logpdf(trialx, *p2), "r-", label="basinhopping")


p3 = pln.fit(sampledata, floc=0, fscale=1.0)
plt.plot(trialx, pln.logpdf(trialx, *p3), "g-", label="MaxLik - built-in.")
plt.legend()
plt.show()
