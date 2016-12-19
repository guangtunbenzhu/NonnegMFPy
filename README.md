NonnegMFPy
=============

Vectorized Nonnegative Matrix Factorization with heteroscedastic uncertainties and missing data


    NonnegMFPy
    ----------

    NonnegMFPy is developed and maintained by Guangtun Ben Zhu, 
    It is designed to solve nonnegative matrix factorization (NMF) given a dataset with 
    heteroscedastic uncertainties and missing data with a vectorized multiplicative 
    update rule (Zhu 2016).

    The un-vectorized (i.e., indexed) update rule for NMF without uncertainties or 
    missing data was originally developed by Lee & Seung (2000), and the un-vectorized 
    update rule for NMF with uncertainties or missing data was originally developed 
    by Blanton & Roweis (2007).

    As all the codes, this code can always be improved and any feedback will be greatly appreciated.

    Note:
      -- Between W and H, which one is the basis set and which one is the coefficient 
         depends on how you interpret the data, because you can simply transpose everything
         as in X-WH versus X^T - (H^T)(W^T)
      -- Everything needs to be non-negative

    Here are some small tips for using this code:

      -- The code can work on multi-dimensional data, such as images, you only need to flatten them to 1D first, 
         e.g., if X.shape = (n_instance, D1, D2)

         >> new_X = X.reshape(n_instance, D1*D2)

         And the code can then work on new_X.


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
         >> chi2_red, time_used = New_g.SolveNMF()

         Caveat: Currently you need to re-instantiate the object whenever you update
         the weight (V), the mask (M), W, H or n_component.
         At the instantiation, the code makes a copy of everything.
         For big jobs with many iterations, this could be a severe bottleneck.
         For now, I think this is a safer way.

      -- It has W_only and H_only options. If you know H or W, and would like
         to calculate W or H. You can run, e.g.,

         >> chi2_red, time_used = g.SolveNMF(W_only=True)

         to get the other matrix (H in this case).

      -- It has sparse mode. However, I found unless the matrix (M or V) is extremely sparse (>90%),
         the speed-up is limited. The overhead of creating the intermediate sparse matrices is small though.
         I haven't made the decision whether the sparse mode should be default.

         >> chi2_red, time_used = g.SolveNMF(sparsemode=True)

      -- One can also change maximum number of iterations or tolerance, e.g.,

         >> chi2_red, time_used = g.SolveNMF(maxiters=5000, tol=1E-7)

         Because W and H are saved internally, you can always re-run SolveNMF (with lower tolerance) 
         without changing maxiters.


    How to Install
    ----------
    I recommend using pip to install the code:
    > pip install NonnegMFPy

    If you are inspired and would like to contribute, you are welcome to clone or fork the repository. 
    Please do drop me a message if you are interested in doing so.


    Test data
    ----------
    
    One can use the following dataset as a test:
      https://s3.us-east-2.amazonaws.com/setcoverproblem/ExtragalacticTest/Extragalatic_Archetype_testsample_spec.fits
      https://s3.us-east-2.amazonaws.com/setcoverproblem/ExtragalacticTest/Extragalatic_Archetype_testsample.fits
    
    The Extragalatic_Archetype_testsample_spec.fits includes spectra (spec: 2760x2820) 
    for 2820 extragalactic sources with 2760 wavelengths 
    and the inverse variance (ivar). 


    Run the test
    -------------
    Assuming you have extracted the spec and ivar matrices from the file above,

   
    Instantiation: 

     >> from NonnegMFPy import nmf
     >> g = nmf.NMF(spec, V=ivar, n_components=5)

    Factorize the data:
     >> chi2, time_used = g.SolveNMF() # Yes, it's that simple!

    If you would like to apply a mask, simply set M in the instantiation:
     >> g = nmf.NMF(spec, V=ivar, M=mask, n_components=5) # Note False means missing datum in M

    If you know W (H) already and would like to learn H (W), you can provide W_known (H_known) 
    and set H_only (W_only) to be true for the projection mode: 

     >> g = nmf.NMF(spec, V=ivar, W=W_known, n_components=5)
     >> chi2, time_used = g.SolveNMF(H_only=True) 

    Note you do need to make sure the dimensions match!

    Output:
        -- The factors are stored in g.W and g.H
        -- The cost can be retrieved with g.cost


Dependencies
=============
    Python >3.5.1
    NumPy >1.11.0
    SciPy >0.17.0 (for sparse matrices)

    I did not fully test the code on earlier versions.

Contact Me
=============
    As all the codes, this code can always be improved and any feedback will be greatly appreciated.


    Sincerely,
    Guangtun Ben Zhu
