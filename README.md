NonnegMFPy
=============

    NonnegMFPy
    ----------

    NonnegMFPy is developed and maintained by Guangtun Ben Zhu, 
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
         the weight (V), the mask (M), W, H or n_component.
         At the instantiation, the code makes a copy of everything.
         For big jobs with many iterations, this could be a severe bottleneck.
         For now, I think this is a safer way.

      -- It has W_only and H_only options. If you know H or W, and would like
         to calculate W or H. You can run, e.g.,

         >> chi2_red, time_used = g.SolveNMF(W_only=True)

         to get the other matrix (H in this case).

    How to Install
    ----------
    I recommend using pip to install the code:
    > pip install NonnegMFPy

    If you are inspired and would like to contribute, you are welcome to clone or fork the repository. 
    Please do drop me a message if you are interested in doing so.


    Test data
    ----------

    Use the Archetype test data:
    The Archetype test data can be retrieved here:
      http://www.pha.jhu.edu/~gz323/scp/Data/ExtragalacticTest/


    Run the test
    -------------
    Input: 
        -- a_matrix[mrows, ncols], the binary relationship matrix
           a_matrix[irow, jcol] = True if jcol covers irow
        -- cost[ncols], the cost of columns. 
           I recommend using normalized cost: cost/median(cost)

    Instantiation: 
        >> a_matrix = np.load('./BeasleyOR/scpa4_matrix.npy')
        >> cost = np.load('./BeasleyOR/scpa4_cost.npy')
        >> g = setcover.SetCover(a_matrix, cost)
    Run the solver: 
        >> solution, time_used = g.SolveSCP()
           ......
           Final Best solution: 234
           Took 1.287 minutes to reach current solution.
           (Results of course will depend on the configuration of your machine)

    Output:
        -- g.s, the (near-optimal) minimal set of columns, a binary 1D array, 
           g.s[jcol] = True if jcol is in the solution
        -- g.total_cost, the total cost of the (near-optimal) minimal set of columns



Dependencies
=============
    Python >3.5.1
    Numpy >1.11.0

    I did not fully test the code on earlier versions.

Contact Me
=============
    As all the codes, this code can always be improved and any feedback will be greatly appreciated.


    Sincerely,
    Guangtun Ben Zhu
