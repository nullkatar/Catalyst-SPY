#!/usr/bin/env python
# coding: utf-8

from mpi4py import MPI
import numpy as np
import math
import os
from random import randint, sample
from sklearn.datasets import load_svmlight_file
import numpy.linalg as lg
import datetime
from scipy.sparse import csr_matrix, eye
from scipy import exp
import sys


n = 47236 #47224
L = 0.251 
mu = 0.001
Lambda2 = 0.001
gam = 2.0 / (L + mu)
Lambda1 = 0.001
x_new = np.ones(n)#np.random.randn(n)
y = []

try: os.remove("./output/X.dat") 
except: pass
try: os.remove("./output/Y.dat") 
except: pass


in_file = open("./output/X.dat", 'a')
in_file.writelines("%s " % place for place in x_new)
in_file.writelines('\n')
in_file.close()

in_file = open("./output/Y.dat", 'a')
y.append(x_new)                                                     #Initialization y0 = x0
in_file.writelines("%s " % place for place in x_new)
in_file.writelines('\n')
in_file.close()

kappa = L - 2 * mu  
betas = []
L = L + kappa
mu = mu + kappa
q = mu / (mu + kappa)
alphas = [math.sqrt(q)]

try: os.remove("./output/GD_func.dat") 
except: pass

MasterFile = "./data/rcv1_train.dat"
Data = load_svmlight_file(MasterFile)
A = csr_matrix(Data[0])
b = csr_matrix(Data[1]).T
m, n = A.shape

def l1(x, lam1):
    return lam1 * np.linalg.norm(x.toarray(), ord = 1)

def f(x, A, b, lam2):
    l = np.log(1 + np.exp(-A.dot(x).multiply(b).A))
    m = b.shape[0]
    return np.sum(l) / m + lam2/2 * np.linalg.norm(x.toarray()) ** 2

def function(x):
    return f(csr_matrix(x).T, A, b, Lambda2) + l1(csr_matrix(x).T, Lambda1)

def f_grad(x_new, A, b, lam2, y, kappa):
    assert ((b.shape[0] == A.shape[0]) & (x_new.shape[0] == A.shape[1]))
    assert lam2 >= 0
    denom = csr_matrix(1/(1 + np.exp(A.dot(x_new).multiply(b).A)))
    g = -(A.multiply(b).multiply(denom).sum(axis=0).T)
    m = b.shape[0]
    return csr_matrix(g) / m + lam2 * x_new + kappa * eye(y.shape[0]).T.dot(eye(y.shape[0]).dot(x_new) - y)

def prox_l1(x, gamma, coef):
    assert(gamma > 0 and coef >= 0)
    lam1 = coef
    return x - abs(x).minimum(lam1 * gamma).multiply(x.sign())

sum = 0
y = [x_new]
for i in range(900):
#while 1 > 0:
    if i > 400:
        sum += x_old  
        
    x_old = x_new
    x_new = prox_l1(csr_matrix(x_old - gam * f_grad(csr_matrix(x_new).T, A, b, Lambda2, csr_matrix(y[-1]).T, kappa).T.toarray()).T, gam, Lambda1).T.toarray()[0]
    
    try: alpha_1 = (q - alphas[-1] ** 2 + math.sqrt((q - alphas[-1] ** 2)**2 + 4 * alphas[-1] ** 2))/2
    except: alpha_1 = -1
    try: alpha_2 = (q - alphas[-1] ** 2 - math.sqrt((q - alphas[-1] ** 2)**2 + 4 * alphas[-1] ** 2))/2
    except: alpha_2 = -1

    if alpha_1 < 1 and alpha_2 < 1 and alpha_1 > 0 and alpha_2 > 0:
        alphas.append(random.choice([alpha_1, alpha_2]))
    elif alpha_1 > 0 and alpha_1 < 1:
        alphas.append(alpha_1)
    elif alpha_2 > 0 and alpha_2 < 1:
        alphas.append(alpha_2)

    betas.append(alphas[-2]  * (1 - alphas[-2] ) / ( alphas[-2]  ** 2 + alphas[-1]))         
    y.append(x_new + betas[-1] * (x_new - x_old))

    out_file = open("./output/GD_func.dat", 'a')
    out_file.writelines("%s " % function(x_new))  
    out_file.writelines('\n')
    out_file.close()

try: os.remove("./data/init.dat") 
except: pass

biba = sum/(900-400) - gam * f_grad(csr_matrix(sum/(900-400)).T, A, b, Lambda2, csr_matrix(y[-1]).T, kappa).T #gam * prox_l1(csr_matrix( f_grad(csr_matrix(sum/(900-400)).T, A, b, Lambda2, csr_matrix(y[-1]).T, kappa).T).T, gam, Lambda1).T.toarray()[0]
biba = biba.tolist()[0]
#print(biba)
in_file = open("./data/init.dat", 'a')
in_file.writelines("%s " % place for place in biba)
in_file.writelines('\n')
in_file.close()
