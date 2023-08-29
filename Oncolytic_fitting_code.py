import pandas as pd
import numpy as np
import scipy
#!pip install scipy==1.7.3
print(scipy.__version__)
from scipy.integrate import odeint
import matplotlib.pyplot as plt
!pip install corner
import corner
from numpy import savetxt
import csv
from google.colab import drive
import pandas as pd

# logistic + immune response + eclipse phase
def lmodelef(y, t, λ, K, β, V, k, δ, p, ε, c, α):
  T, E, I, V, F = y
  DTdt = λ * T * (1 - ((T + E + I)/K)) - β * T * V
  DEdt = β * T * V - k * E
  DIdt = k * E - δ * I
  DVdt = (p / (1 + ε * F)) * I - c * V
  DFdt = V - α * F
  return DTdt, DEdt, DIdt, DVdt, DFdt

# exponential + eclipse + immune
def emodelef(y, t, β, V, k, δ, p, ε, c, α):
  T, E, I, V, F = y
  DTdt = 0.09959108 * T - β * T * V
  DEdt = β * T * V - k * E
  DIdt = k * E - δ * I
  DVdt = (p / (1 + ε * F)) * I - c * V
  DFdt = V - α * F
  return DTdt, DEdt, DIdt, DVdt, DFdt

#K
def basemodel(T, t, λ, K):
  DTdt = λ * T * (1 - ((T)/K))
  return DTdt

# none
def basemodelnk(T, t, λ, K):
  DTdt = λ * T
  return DTdt
  
def PBSSSR(y):
  λ, K = y[0], y[1]
  ary = ((odeint(basemodel, PBSy[0], PBSx, args = (λ, K)).reshape(15,) - PBSy)**2)
  SSR = sum(ary)
  return SSR

PBSfit = scipy.optimize.minimize(PBSSSR, [0.123, 1000], method = 'Nelder-Mead')
print(PBSfit)

PBSres = (odeint(basemodel, [PBSy[0]], PBSx,
                        args= (PBSfit.x[0], PBSfit.x[1])))

plt.scatter(PBSx, PBSy, s = 8, color='orangered', label = "Actual measurement")
plt.plot(PBSx, PBSres, color='orangered',label = "Best fit line")
plt.title(label = "Volume of untreated tumor over time")
plt.ylabel('Tumor Volume (mm^3)')
plt.xlabel('Time (days)')
plt.legend()
print(PBSfit)

def Ad1SSRefl(y):
  λ, K, β, V, k, δ, p, ε, c, α = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]
  Ad1res = np.sum(odeint(lmodelef, [Ad1y[0], 0, 0, y[3], 0], Ad1x, args = (λ, K, β, V, k, δ, p, ε, c, α))[:,0:3], axis = 1)
  # ary = ((odeint(basemodel, PBSy[0], PBSx, args = (λ, K)).reshape(15,) - PBSy)**2)
  # SSR1 = sum(ary)
  SSR = sum((Ad1res - (Ad1y))**2)
  return SSR

Ad1fit = scipy.optimize.minimize(Ad1SSRefl, [1.21264372e-01, 1.34682793e+03, 1.39195274e+01, 1.48402225e-02,
       8.03176782e-02, 8.03308027e-02, 2.53750468e-04, 4.34801695e+01,
       1.32574167e-01, 4.06501041e-08], method = 'Nelder-Mead', bounds = ([[0, None], [0, None],[0, None], [0, None],
                                                           [0, None],[0.01, None], [0.00, None], [0.00, None], [0.00, None], [0.00, None]]))


print(Ad1fit)
print(Ad1fit.fun - PBSSSR([Ad1fit.x[0], Ad1fit.x[1]]))
Ad1res = np.sum((odeint(lmodelef, [Ad1y[0], 0, 0, Ad1fit.x[3], 0], Ad1x,
                        args= (Ad1fit.x[0], Ad1fit.x[1],Ad1fit.x[2],Ad1fit.x[3],
                               Ad1fit.x[4], Ad1fit.x[5], Ad1fit.x[6], Ad1fit.x[7], Ad1fit.x[8], Ad1fit.x[9])))[:,0:3], axis = 1)

plt.scatter(Ad1x, Ad1y, s = 8, color='royalblue', label = "Actual measurement")
plt.plot(Ad1x, Ad1res, color='royalblue',label = "Best fit line")
plt.title(label = "Volume of tumor treated with Ad1d24.P19 virus over time")
plt.ylabel('Tumor Volume (mm^3)')
plt.xlabel('Time (days)')
plt.legend()

#resid immune
residAd1 = np.sum((odeint(emodelef, [Ad1y[0], 0, 0,  8.79257095e-03,0], Ad1x, args = (1.45082670e+01, 8.79259920e-03, 7.40647857e-02, 7.40642482e-02,
       1.24299398e-04, 4.75767503e+01, 6.85745311e-02, 4.94808468e-13
)))[:, 0:3], axis=1).reshape(42) - Ad1y

Ad1res = np.sum((odeint(emodelef, [Ad1y[0], 0, 0,  8.79257095e-03, 0], Ad1x,
                        args= (1.45083051e+01, 8.79257095e-03, 7.40643144e-02, 7.40644832e-02,
       1.24300654e-04, 4.75773518e+01, 6.85744470e-02, 4.85400501e-13
)))[:,0:3], axis = 1)

def Ad1SSRbs(y):
  β, V, k, δ, p, ε, c, α = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]
  Ad1fit = np.sum(odeint(emodelef, [Ad1y[0], 0, 0, y[3], 0], Ad1x, args = (β, V, k, δ, p, ε, c, α))[:,0:3], axis = 1)
  SSR = np.sum((Ad1fit - (shuffled_errorAd1))**2)
  return SSR

#bootstrapping immune
res_arrayAd1 = np.empty((8,))
ssraryAd1 = np.empty((1,))

for x in range(1, 1001):
  resad1 = np.random.choice(residAd1, 42)
  shuffled_errorAd1 = resad1 + Ad1res
  res_parameters = scipy.optimize.minimize(Ad1SSRbs, [1.45083051e+01, 8.79257095e-03, 7.40643144e-02, 7.40644832e-02,
       1.24300654e-04, 4.75773518e+01, 6.85744470e-02, 4.85400501e-13], method = 'Nelder-Mead', bounds = ([[0, None],
                                                           [0, None], [ 0.01, None], [0, None], [0, None], [0, None], [0, None], [0, None]]))
  res_arrayAd1 = np.vstack((res_arrayAd1, res_parameters.x))
  ssraryAd1 = np.vstack((ssraryAd1, res_parameters.fun))

res_arrayAd1 = np.delete(res_arrayAd1, 0, axis = 0)
ssraryAd1 = np.delete(ssraryAd1, 0, axis=0)

ssraryAd1 = np.sort(ssraryAd1, axis = 0)

# lambda_vals = np.sort(res_arrayAd1[:, 0])
# K_vals = np.sort(res_arrayAd1[:, 1])
beta_vals = np.sort(res_arrayAd1[:, 0])
V_vals = np.sort(res_arrayAd1[:, 1])
k_vals = np.sort(res_arrayAd1[:, 2])
delta_vals = np.sort(res_arrayAd1[:, 3])
p_vals = np.sort(res_arrayAd1[:, 4])
epsilon_vals = np.sort(res_arrayAd1[:, 5])
c_vals = np.sort(res_arrayAd1[:, 6])
alpha_vals = np.sort(res_arrayAd1[:, 7])


print(ssraryAd1[24], ssraryAd1[974])


print(beta_vals[24], beta_vals[974])
print(V_vals[24], V_vals[974])
print(k_vals[24], k_vals[974])
print(delta_vals[24], delta_vals[974])
print(p_vals[24], p_vals[974])
print(epsilon_vals[24], epsilon_vals[974])
print(c_vals[24], c_vals[974])
print(alpha_vals[24], alpha_vals[974])

labels = ['β', 'V', 'k', 'δ', 'p','ε', 'c', 'α']
figure = corner.corner(res_arrayAd1, labels = labels, show_titles = True)

