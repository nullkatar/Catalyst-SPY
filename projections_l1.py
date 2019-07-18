import numpy as np
from scipy.sparse import csr_matrix, eye, coo_matrix

# This file is for basic coordinate-wise projections functions
# To create a projection set necessary input is:
#  - n - space dimension
#  - k - amount of nonzero coordinates (measure of sparsity)
#  - type - adaptive or not
#  - pattern (only for adaptive)

# Expected projection will be a diagonal matrix with entries of:
# - k/n in nonadaptive case
# - 1 for support and k/(n-s) for adaptive where s is a support size


def AdjointSetL1(dim, coord_set):
    """
    :param dim: dimension of the full space
    :param coord_set: set to find adjoint to
    :return: sorted set of coordinates that are not in coord_set
    """
    all_coords = range(dim)
    if len(coord_set) == 0:
        return all_coords
    
    i1 = 0
    i2 = 0
    res = []
    while (i1 < dim and i2 < len(coord_set)):
        if all_coords[i1] != coord_set[i2]:
            res.append(all_coords[i1])
            i1+=1
        else:
            i1+=1
            i2+=1
    return res
    
#    for elem in coord_set:
#        all_coords.remove(elem)
#    return all_coords

def SparsitySetL1(x):
#    xx = x.toarray()
#    res = []
#    for i in range(len(xx)):
#        if xx[i] != 0:
#            res.append(i)
#    return res
    return list(x.nonzero()[0])
    

class projection:
    def __init__(self, n, k, tpe = 'usual', pattern = []):
        self.n = n
        self.k = k
        self.type = tpe
        self.pattern = pattern
        self.adj_pattern = AdjointSetL1(n, pattern)
        self.ComputePbar()
        
    def SparsitySet(self, x):
        return SparsitySetL1(x)

# global functions
    
# local class functions
    def RandomSubset(self, target_set):
        """
        :param target_set: set to select from
        :return: if size > size of target set - return target set, else return sorted subset of size "size"
        """
        if (self.k >= len(target_set)):
            return target_set
        return np.sort(list(np.random.choice(target_set, self.k, replace = False)))
    
    
    def ComputePbar(self):
        if self.type == 'usual':
            res = self.k/(self.n+0.0)*eye(self.n)
        else:
            row = list(range(self.n))
            column = list(row)
            data = np.zeros(self.n)
            nmk = self.n - len(self.pattern) - 0.0
            for coord in self.pattern:
                data[coord] = 1
            for coord in self.adj_pattern:
                data[coord] = self.k / nmk
            res = csr_matrix((data, (row, column)), shape = (self.n, self.n))
        self.pbar = res
        self.qbar = eye(self.n)
        self.qbar_inv = eye(self.n)
    
    
    def ChangePattern(self, x):
        self.pattern = self.SparsitySet(x)
        self.adj_pattern = AdjointSetL1(self.n, self.pattern)
        self.ComputePbar()

        
     
    def ChangeK(self, newk):
        self.k = newk
        self.ComputePbar()
        
    
    def ProjectionBySet(self, coord_set):
        data = np.ones(len(coord_set))
        return csr_matrix((data, (coord_set, coord_set)), shape = (self.n, self.n))

    def RandomProjection(self):
        """
        :return: a random projection with respect to type, if adaptive, uses current sparsity pattern
        """
        if  self.type == 'usual':
            rand_set = self.RandomSubset(range(self.n))
            return self.ProjectionBySet(rand_set)
        rand_set = list(self.RandomSubset(self.adj_pattern))
        coord_set = rand_set
        if len(self.pattern) > 0:
            coord_set = coord_set + self.pattern
            coord_set.sort()
        res = self.ProjectionBySet(coord_set)
        return res