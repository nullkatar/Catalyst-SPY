#!/usr/bin/env python
import sys
import os
import math
import numpy as np
from scipy.sparse import csr_matrix

def prox_l1(x, gamma, coef):
    assert(gamma > 0 and coef >= 0)
    lam1 = coef
    return x - abs(x).minimum(lam1 * gamma).multiply(x.sign())


################################
###########Constants############
################################

y = []
n = 47236 
num_iters = 1000
L = 0.251 
mu = 0.001
Lambda1 = 0.001
gam = 2.0 / (L + mu)


kappa_1 = (-(mu + 3*L) + math.sqrt((mu+3*L)**2 + 8*(L*L+2*mu*mu-5*mu*L))) / 4
kappa_2 = (-(mu + 3*L) - math.sqrt((mu+3*L)**2 + 8*(L*L+2*mu*mu-5*mu*L))) / 4
kappa = max(kappa_1, kappa_2)                                                                               #What to choose?
q = mu / (mu + kappa)
L = 0.251 + kappa
mu = 0.001 + kappa
betas = []
alphas = [math.sqrt(q)]

################################
##########Initialization########
################################
os.system('python GD.py')
print('========================================END OF GD======================================================')

try: os.remove("./output/times.dat"),
except: pass
try: os.remove("./output/func.dat"),
except: pass
try: os.remove("./output/someshit.dat"),
except: pass
try: os.remove("./output/exchangest.dat"),
except: pass

# barx = np.random.randn(n)                                                                              #First initialization
# in_file = open("./output/X.dat", 'a')
# in_file.writelines("%s " % place for place in barx)
# in_file.writelines('\n')
# in_file.close()

# in_file = open("./output/Y.dat", 'a')
# y.append(barx)#y.append(prox_l1(csr_matrix(barx).T, gam, Lambda1).T.toarray()[0])                      #Initialization y0 = x0
# in_file.writelines("%s " % place for place in y[0])
# in_file.writelines('\n')
# in_file.close()

X =  open("./output/X.dat", 'r')
Y =  open("./output/Y.dat", 'r')
barx = np.array([float(iter) for iter in (list(X))[0].split()]) #np.random.randn(n)
y_s = np.array([float(iter) for iter in (list(Y))[0].split()])
X.close()
Y.close()

#################################
##########Catalyst###############
#################################

for k in range(num_iters):
    func =[500]
    if k != 0:
        in_file = open("./output/func.dat", 'r')
        contents = in_file.readlines()
        func = [float(bun) for bun in contents[-1].split()]                                                    #Reading X
        in_file.close()    
    
    #if func[0] - 0.6192333812543935 > 1e-1:
    os.system('mpiexec -np 10 ./go_19.py')
    #else:
    #    os.system('mpiexec -np 10 ./go_new.py {}'.format(k+1))
    print('=============================================={} ITERATIONS OF CATALYST DONE=========================================='.format(k+1))
    in_file = open("./output/X.dat", 'r')
    contents = in_file.readlines()
    x_prev = [float(bun) for bun in contents[-2].split()]                                               #Reading previous X
    x = [float(bun) for bun in contents[-1].split()]                                                    #Reading X
    in_file.close()
    
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

    betas.append(alphas[-2]  * (1 - alphas[-2] ) / ( alphas[-2]  ** 2 + alphas[-1]  ))         
    y.append(x + betas[-1] * (np.array(x) - np.array(x_prev)))
    
    
    out_file = open("./output/Y.dat", 'a')
    out_file.writelines("%s " % place for place in y[-1])                                                      #Writing Y
    out_file.writelines('\n')
    out_file.close()
