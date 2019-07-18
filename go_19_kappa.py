#!/usr/bin/env python

from mpi4py import MPI
import numpy as np
import time
import sys
import os
from random import randint, sample
from sklearn.datasets import load_svmlight_file
import numpy.linalg as lg
import datetime
from scipy.optimize import minimize
from projections_l1 import *
from functions import *
import math

###### Assumption on namings
###### Each worker has a datafile of the following form

#####     dataset_worker_workernumber.dat



#####     each algorithm generates a file with results of the form
#####     algorithm_dataset_amountOfWorkers.dat




##################################################
##################################################
####  INITIALIZATION
##################################################
##################################################


def function(x):
    return f(csr_matrix(x).T, A, b, Lambda2)+l1(csr_matrix(x).T, Lambda1)

##################################################
####  MPI
##################################################

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
info = MPI.Status()
SlaveAmount = size - 1

#################################################
####  PROBLEM PARAMETERS
##################################################

#Problem = "Lasso"
Problem = "Logistic"
#Mask = "Full"
#Mask    = "Part"
num_coord = 5000
Mask    = "Mask_Part"
#Mask    = "Mask_Ada"
#Mask    = "Mask_Cheat"
#ProblemFile = "Lasso_Simple"

ProblemFile = "rcv1_train"
Lambda1 = 0.001
ITE_MAX = 5
Debug = True

#################################################
####  PROBLEM FILES
##################################################

if rank == 0:
    MasterFile = "./data/" + ProblemFile + ".dat"
else:
    SlaveFile = "./data/slaves/" + ProblemFile + "_worker_" + str(rank) + ".dat"

comm.Barrier()

#################################################
#### INITIALIZING DATA
##################################################

n = 0

if rank == 0:
    Data = load_svmlight_file(MasterFile)
    A = csr_matrix(Data[0])
    b = csr_matrix(Data[1]).T
    m, n = A.shape

n = comm.bcast(n, root=0)

if rank != 0:
    Data = load_svmlight_file(SlaveFile)
    A = Data[0].toarray()
    b = csr_matrix(Data[1]).T
    m, barn = A.shape
    A = np.hstack((A, np.zeros((m, n - barn))))
    A = csr_matrix(A)

pgather = comm.gather(m, root=0)

if rank == 0:
    proportions = np.array(pgather[1:])/float(pgather[0])
    print("[MASTER] Propotions: {}".format(proportions))

comm.Barrier()

#################################################
#### CONSTANTS COMPUTATIONS
#################################################

m_tot = comm.bcast(m, root = 0)
Lambda2 = 1.0/m_tot


if Problem == "Logistic":
    L = 0
    mu = np.inf
    if rank != 0:
        L = 0.25 * np.max(A.multiply(A).sum(axis = 1)) + Lambda2
        mu = Lambda2
        if Debug:
            print("[SLAVE {}] Computed L = {} and mu = {}".format(rank, L, mu))
    L = comm.allreduce(L, MPI.MAX)
    mu = comm.allreduce(mu, MPI.MIN)

    if rank != 0 and Debug:
        print("[SLAVE {}] I have L = {} and mu = {}".format(rank, L, mu))

gam = 0.0
if rank == 0:
    gam = 2.0 / (L + mu)

comm.Barrier()
gam = comm.bcast(gam, root = 0)

kappa_1 = (-(mu + 3*L) + math.sqrt((mu+3*L)**2 + 8*(L*L+2*mu*mu-5*mu*L))) / 4
kappa_2 = (-(mu + 3*L) - math.sqrt((mu+3*L)**2 + 8*(L*L+2*mu*mu-5*mu*L))) / 4
kappa_3 = 19*L - 20*mu
kappa = max(kappa_1, kappa_2, kappa_3)

mu = mu + kappa
L = L + kappa
comm.Barrier()

#################################################
#################################################
#### MAIN ITERATION
#################################################
#################################################

if rank == 0:
    print("[MASTER] Beginning main iteration")
    print("[MASTER] Problem: {}      Size: {}x{}".format(Problem, m, n))
    print("[MASTER] Conditionning: mu = {}  L = {}".format(mu, L))

##################################################
####  STEPSIZE AND INIT BROADCAST
##################################################

if rank != 0 and Debug:
    print("[SLAVE {}] Stepsize: gamma = {}".format(rank, gam))
if rank == 0:
    print("[MASTER] Stepsize: gamma = {}".format(gam))

barx = np.zeros(n)

if rank == 0:
    k = 0
    StartTime = time.time()
    X =  open("./output/X_kappa.dat", 'r')
    Y =  open("./output/Y_kappa.dat", 'r')
    k = 0
    barx = np.array([float(iter) for iter in (list(X)[-1]).split()]) #np.random.randn(n)
    y_s = np.array([float(iter) for iter in (list(Y)[-1]).split()])
    X.close()
    Y.close()
comm.Barrier()

comm.Bcast(y_s, root = 0)
comm.Bcast(barx, root = 0)

comm.Barrier()
##################################################
####  MAIN
##################################################


buf_ind = []
buf_val = []

if rank == 0:
    iter_tab = []  # iterations
    val_tab = []  # value
    sp_tab = []  # sparsity
    ex = 0
    ex_tab = []  # number of coordinates exchanged
    time_tab = []  # time
    startTime = time.time()

Up = True
FirstTime = True

while Up:
    
    ##################################################
    ####  MASTER
    ##################################################
    
    if rank == 0:
        ms = np.zeros(size)
        ms_counter = 0

        with open('./data/init.dat') as file:
            lines = file.readlines()
        file.close()
        vect = np.linalg.norm(barx - [float(plebs) for plebs in lines[0].split()]) ** 2

        with open('./output/GD_func.dat') as file:
            bepis = file.readlines()
        file.close()
        bepis = [float(plebs) for plebs in bepis]

        while k < ITE_MAX:
            k += 1
            buf_ind = comm.recv(source = MPI.ANY_SOURCE, status = info)
            sender = info.Get_source()

            ms[sender] += 1
            for i in range(len(ms)):
                if ms[i] > 2:
                    ms_counter += 1
                    ms = np.zeros(size)

			buf_val = np.array(comm.recv(source = sender))
            barx[buf_ind] = barx[buf_ind] + proportions[sender - 1] * buf_val
            
            # print("[MASTER] barx = {} after sender {}".format(barx,sender) )
            
            x = prox_l1(csr_matrix(barx).T, gam, Lambda1).T.toarray()[0]
            
            buf_ind_out = list(np.flatnonzero(x))
            if len(buf_ind_out) == 0:
                buf_ind_out = [0]

            buf_val = list(x[buf_ind_out])
            comm.send(buf_ind_out, dest = sender)
            comm.send(buf_val, dest = sender)
            
            ex += len(buf_ind) + len(buf_ind_out)
            
            #if k % 100 == 0:
            print("[MASTER] {} ".format(k))
            iter_tab.append(k)
            val_tab.append(x)
            sp_tab.append(len(buf_ind_out) * 100.0 / float(n))
            ex_tab.append(ex)
            time_tab.append(time.time() - startTime)
            if k == 18:
                print(function(val_tab[-1]))

        slaves_stopped = 0
        
        while slaves_stopped < SlaveAmount:
            slaves_stopped += 1
            buf_ind = comm.recv(source = MPI.ANY_SOURCE, status = info)
            sender = info.Get_source()
            buf_val = np.array(comm.recv(source = sender))
            buf_ind = [-1]
            comm.send(buf_ind, dest = sender)
            comm.send(buf_val, dest = sender)
            print("[MASTER] Stopped Slave {} ({}/{})".format(sender, slaves_stopped, SlaveAmount))
        
        Up = False

    ##################################################
    ####  SLAVES
    ##################################################
    
    else:
               
	if FirstTime:
            FirstTime = False
            x_i = np.copy(barx)
            buf_ind = list(range(n))
            buf_val = np.copy(x_i)
        
        else:
            
            buf_ind = comm.recv(source = 0)
            if len(buf_ind) == 0:
                print("[SLAVE {}] NOTHING RECEIVED!".format(rank))
            buf_val = np.array(comm.recv(source = 0))
            
            if buf_ind[0] == -1:
                print("[SLAVE {}] Stopping".format(rank))
                Up = False
                break
            
            barx = np.zeros(n)
            barx[buf_ind] = buf_val
        
        if not Up:
            break

        x_i_old = np.copy(x_i)
        L_recv = len(buf_ind)
        
        if Mask == "Full":
            S = list(range(n))
        
        if Mask == "Part":
            S = np.random.choice(range(n), num_coord, replace = False)
        
        if Mask == "Mask_Part" or Mask == "Mask_Cheat":
#            S_bar = [i for i in list(range(n)) if i not in buf_ind]
			p = 0.1            
			S_bar = AdjointSetL1(n, buf_ind)
            n_bar = min(len(S_bar), math.ceil(p*len(num_coord)))
            
            if len(S_bar) == 0:
                p = 1.0
                S = buf_ind
            
            else:
                p = n_bar / float(len(S_bar))
                S = buf_ind + list(np.random.choice(S_bar, n_bar, replace = False))
        
        if Mask == "Mask_Ada":
            S_bar = AdjointSetL1(n, buf_ind)
#            S_bar = [i for i in list(range(n)) if i not in buf_ind]
            n_bar = min(len(S_bar), 5 * L_recv)
            
            if len(S_bar) == 0:
                p = 1.0
                S = buf_ind
            
            else:
                p = n_bar / float(len(S_bar))
                S = buf_ind + list(np.random.choice(S_bar, n_bar, replace = False))
        
        x_i[S] = barx[S] - gam * f_grad(csr_matrix(barx).T, A, b, Lambda2, csr_matrix(y_s).T, kappa).T.toarray()[0][S]
        
        #if Mask == "Mask_Part" or Mask == "Mask_Ada":
            #x_i[S] = p * x_i[S] + (1 - p) * x_i_old[S]
        
        # print("[SLAVE {}] x_i = {}".format(rank,x_i))
        
        Delta = x_i - x_i_old
        buf_ind = S
        buf_val = list(Delta[buf_ind])
        comm.send(buf_ind, dest = 0)
        comm.send(buf_val, dest = 0)

comm.Barrier()

if rank == 0:
    print("[MASTER] ITERATIONS DONE")


#################################################
#################################################
#### RESULTS
#################################################
#################################################

if rank == 0:
    out_file = open("./output/X_kappa.dat", 'a')
    out_file.writelines("%s " % place1 for place1 in x)                                                   #Writing X
    out_file.writelines('\n')
    out_file.close()

    out_file = open("./output/times_kappa.dat", 'a')
    out_file.writelines("%s " % place2 for place2 in time_tab)                                            #Writing Times
    out_file.writelines('\n')
    out_file.close()

    out_file = open("./output/func_kappa.dat", 'a')
    for i in range(len(val_tab)):
        out_file.writelines("%s " %  function(val_tab[i]))                                                #Writing Obj Function
        out_file.writelines('\n')
    out_file.close()

    out_file = open("./output/exchanges_kappa.dat", 'a')
    out_file.writelines("%s " % place2 for place2 in ex_tab)                                              #Writing Exchanges
    out_file.writelines('\n')
    out_file.close()

#################################################
#################################################
#### FINISH
#################################################
#################################################

comm.Barrier()

if rank == 0:
    print("[MASTER] FINISHED")
