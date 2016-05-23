__license__ = "MIT"
__author__ = "Guangtun Ben Zhu (BGT) @ Johns Hopkins University"
__startdate__ = "2014.11.30"
__name__ = "nmf"
__module__ = "NonnegMFPy"
__python_version__ = "3.5.1"
__numpy_version__ = "1.11.0"
__scipy_version__ = "0.17.0"

__lastdate__ = "2016.05.22"
__version__ = "0.1.0"


__all__ = ['NMF']

""" 
nmf.py

    This piece of software is developed and maintained by Guangtun Ben Zhu, 
    It is designed to solve nonnegative matrix factorization (NMF) given a dataset with heteroscedastic 
    uncertainties and missing data with a vectorized multiplicative update rule (Zhu 2016).
    The un-vectorized (i.e., indexed) update rule for NMF without uncertainties or missing data was
    originally developed by Lee & Seung (2000), and the un-vectorized update rule for NMF
    with uncertainties or missing data was originally developed by Blanton & Roweis (2007).

    As all the codes, this code can always be improved and any feedback will be greatly appreciated.

    Note:
      -- Between W and H, which one is the basis set and which one is the coefficient
         depends on how you interpret the data, because you can simply transpose everything
         as in X-WH versus X^T - (H^T)(W^T)
      -- Everything needs to be non-negative

    Here are some small tips for using this code:
      -- The algorithm can handle heteroscedastic uncertainties and missing data.
         You can supply the weight (V) and the mask (M) at the instantiation:

         >> g = nmf.NMF(X, V=V, M=M)

         This can also be very useful if you would like to iterate the process
         so that you can exclude certain new data by updating the mask.
         For example, if you want to perform a 3-sigma clipping after an iteration
         (assuming V is the inverse variance below):

         >> chi2_red, time_used = g.SolveNMF()
         >> New_M = np.copy(M)
         >> New_M[np.fabs(np.sqrt(V)*(X-np.dot(g.W, g.H)))>3] = False
         >> New_g = nmf.NMF(X, V=V, M=New_M)

         Caveat: Currently you need to re-instantiate the object whenever you update
         the weight (V), the mask (M), W, H or n_components.
         At the instantiation, the code makes a copy of everything.
         For big jobs with many iterations, this could be a severe bottleneck.
         For now, I think this is a safer way.

      -- It has W_only and H_only options. If you know H or W, and would like
         to calculate W or H. You can run, e.g.,

         >> chi2_red, time_used = g.SolveNMF(W_only=True)

         to get the other matrix (H in this case).


    Copyright (c) 2015-2016 Guangtun Ben Zhu

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
    and associated documentation files (the "Software"), to deal in the Software without 
    restriction, including without limitation the rights to use, copy, modify, merge, publish, 
    distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or 
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
    BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


import numpy as np
from time import time

# Some magic numbes
_largenumber = 1E100
_smallnumber = 1E-5

class NMF:
    """
    Nonnegative Matrix Factorization - Build a set of nonnegative basis components given 
    a dataset with Heteroscedastic uncertainties and missing data with a vectorized update rule.

    Algorithm:
      -- Iterative multiplicative update rule

    Input: 
      -- X: m x n matrix, the dataset

    Optional Input/Output: 
      -- n_components: desired size of the basis set, default 5
      -- tol: convergence criterion, default 1E-5
      -- maxiters: allowed maximum number of iterations, default 1000

      -- V: m x n matrix, the weight, (usually) the inverse variance
      -- M: m x n binary matrix, the mask, False means missing/undesired data
      -- H: n_components x n matrix, the H matrix, usually interpreted as the coefficients
      -- W: m x n_components matrix, the W matrix, usually interpreted as the basis set

    (Use A.4 instance from Beasley's OR library,
     http://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html, as an example)
    
    Construct a new basis set with 12 components
    Instantiation: 
        >> g = nmf.NMF(flux, V=ivar, n_components=12)
    Run the solver: 
        >> chi2_red, time_used = g.SolveNMF()

    If you have a basis set, say W, you would like to calculate the coefficients:
        >> g = nmf.NMF(flux, V=ivar, W=W_known)
    Run the solver: 
        >> chi2_red, time_used = g.SolveNMF(H_only=True)

    Comments:
      -- Between W and H, which one is the basis set and which one is the coefficient 
         depends on how you interpret the data, because you can simply transpose everything
         as in X-WH versus X^T - (H^T)(W^T)
      -- Everything needs to be non-negative

    References:
      -- Guangtun Ben Zhu, 2016
         A Vectorized Algorithm for Nonnegative Matrix Factorization with 
         Heteroskedastic Uncertainties and Missing Data
         AJ/PASP, (to be submitted)
      -- Blanton, M. and Roweis, S. 2007
         K-corrections and Filter Transformations in the Ultraviolet, Optical, and Near-infrared
         The Astronomical Journal, 133, 734
      -- Lee, D. D., & Seung, H. S., 2001
         Algorithms for non-negative matrix factorization
         Advances in neural information processing systems, pp. 556-562

    To_do:
      -- 

    History:
        -- 22-May-2016, Documented, BGT, JHU
        -- 13-May-2016, Add projection mode (W_only, H_only), BGT, JHU
        -- 30-Nov-2014, Started, BGT, JHU
    """

    def __init__(self, X, W=None, H=None, V=None, M=None, n_components=5, maxiters=1000, tol=_smallnumber):
        """
        Initialization
        
        Required Input:
          X -- the input data set

        Optional Input/Output:
          -- n_components: desired size of the basis set, default 5
          -- tol: convergence criterion, default 1E-5
          -- maxiters: allowed maximum number of iterations, default 1000

          -- V: m x n matrix, the weight, (usually) the inverse variance
          -- M: m x n binary matrix, the mask, False means missing/undesired data
          -- H: n_components x n matrix, the H matrix, usually interpreted as the coefficients
          -- W: m x n_components matrix, the W matrix, usually interpreted as the basis set
        """

        # I'm making a copy for the safety of everything; should not be a bottleneck
        self.X = np.copy(X) 
        if (np.count_nonzero(self.X<0)>0):
            print("There are negative values in X. Setting them to be zero...", flush=True)
            self.X[self.X<0] = 0.

        self.n_components = n_components
        self.maxiters = maxiters
        self.tol = tol

        if (W is None):
            self.W = np.random.rand(self.X.shape[0], self.n_components)
        else:
            if (W.shape != (self.X.shape[0], self.n_components)):
                raise ValueError("Initial W has wrong shape.")
            self.W = np.copy(W)
        if (np.count_nonzero(self.W<0)>0):
            print("There are negative values in W. Setting them to be zero...", flush=True)
            self.W[self.W<0] = 0.

        if (H is None):
            self.H = np.random.rand(self.n_components, self.X.shape[1])
        else:
            if (H.shape != (self.n_components, self.X.shape[1])):
                raise ValueError("Initial H has wrong shape.")
            self.H = np.copy(H)
        if (np.count_nonzero(self.H<0)>0):
            print("There are negative values in H. Setting them to be zero...", flush=True)
            self.H[self.H<0] = 0.

        if (V is None):
            self.V = np.ones(self.X.shape)
        else:
            if (V.shape != self.X.shape):
                raise ValueError("Initial V(Weight) has wrong shape.")
            self.V = np.copy(V)
        if (np.count_nonzero(self.V<0)>0):
            print("There are negative values in V. Setting them to be zero...", flush=True)
            self.V[self.V<0] = 0.

        if (M is None):
            self.M = np.ones(self.X.shape, dtype=np.bool)
        else:
            if (M.shape != self.X.shape):
                raise ValueError("M(ask) has wrong shape.")
            if (M.dtype != np.bool):
                raise TypeError("M(ask) needs to be boolean.")
            self.M = np.copy(M)

        # Set masked elements to be zero
        self.V[(self.V*self.M)<=0] = 0
        self.V_size = np.count_nonzero(self.V)

    @property
    def cost(self):
        """
        Total cost of a given set s
        """
        diff = self.X - np.dot(self.W, self.H)
        chi2 = np.einsum('ij,ij', self.V*diff, diff)/self.V_size
        return chi2

    def SolveNMF(self, W_only=False, H_only=False):
        """
        Construct the NMF basis

        Keywords:
            -- W_only: Only update W, assuming H is known
            -- H_only: Only update H, assuming W is known
               -- Only one of them can be set

        Output: 
            -- chi2: final cost


        """

        t0 = time()


        if (W_only and H_only):
            print("Both W_only and H_only are set to be True. Return ...", flush=True)
            return (chi2, time_used)

        chi2 = self.cost
        oldchi2 = _largenumber

        XV = self.X*self.V

        niter = 0

        while (niter<self.maxiters) and ((oldchi2-chi2)/oldchi2 > self.tol):

            # Update H
            if (not W_only):
                H_up = np.dot(self.W.T, XV)
                H_down = np.dot(self.W.T, np.dot(self.W, self.H)*self.V)
                self.H = self.H*H_up/H_down

            # Update W
            if (not H_only):
                W_up = np.dot(XV, self.H.T)
                W_down = np.dot(np.dot(self.W, self.H)*self.V, self.H.T)
                self.W = self.W*W_up/W_down

            # chi2
            oldchi2 = chi2
            chi2 = self.cost

            # Some quick check. May need its error class ...
            if (not np.isfinite(chi2)):
               raise ValueError("NMF construction failed, likely due to missing data")

            if (np.mod(niter, 20)==0):
                print("Current Chi2={0:.4f}, Previous Chi2={1:.4f}, Change={2:.4f}% @ niters={3}".format(chi2,oldchi2,(oldchi2-chi2)/oldchi2*100.,niter), flush=True)

            niter += 1
            if (niter == self.maxiters):
                print("Iteration in re-initialization reaches maximum number = {0}".format(niter), flush=True)

        time_used = (time()-t0)/60.
        print("Took {0:.3f} minutes to reach current solution.".format(time_used), flush=True)

        return (chi2, time_used)

