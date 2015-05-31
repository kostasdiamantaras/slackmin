# Slackmin Classification Algorithm with example.

This is a MATLAB implementation of the plain vanilla Slackmin algorithm presented in the paper

[M. Kotti, K. I. Diamantaras, "Efficient Binary Classification Through Energy Minimisation of Slack Variables",
Neurocomputing, Elsevier, Vol. 148, pp. 498–511, DOI: 10.1016/j.neucom.2014.07.013, January 2015] (http://www.sciencedirect.com/science/article/pii/S0925231214008911)

This implementation does not contain the search procedures for finding the optimal hyper-parameters.

### Files:

----------

#### [slackmin_train.m](https://github.com/kostasdiamantaras/slackmin/blob/master/slackmin_train.m)

Matlab function implementing the *Slackmin* training algorithm.

Usage: *[model, y, accuracy] = slackmin_train(x, t, params)*

```  
   x = [nxP] pattern matrix
       (n = pattern dimension, P = number of patterns)
   t = [1xP] target vector (values = -1/1)
   params = struct(
     'kernel', (values = 'linear'(default), 'rbf', 'poly'), ...
     'BASIS_SIZE', (values = interger between 1 and P, default = min(100,P)), ...
     'MAXEPOCHS', (values = number of training epochs, default = 20), ...
     % parameter gamma, for RBF kernel  K(x,y) = exp(-gamma * norm(x-y)^2):
     'gamma', (values = positive double, default = 1), ...
     % parameters theta, d, for Polynomial kernel  K(x,y) = (x'*y + theta)^d:
     'theta', (values = double, default = 1), ...
     'd', (values = integer, default = 2) ...
   )
```

Returns:

```
   y = output vector (values = double).
       Ideally (y>0) if t=+1,  (y<0) if t=-1
   accuracy = classification accuracy (value = double between 0 and 100)
```

-----------

#### [slackmin_sim.m](https://github.com/kostasdiamantaras/slackmin/blob/master/slackmin_sim.m)

Matlab function implementing the *Slackmin* recall (after the model has been trained).

Usage: *[y, accuracy] = slackmin_sim(x, t, model)*
```
   x = [nxP] pattern matrix
       (n = pattern dimension, P = number of patterns)
   t = [1xP] target vector (values = -1/1)
   model = srtuct containing model parameters created by slackmin_train()
```


Returns:
```
   y = output vector (values = double).
       Ιdeally (y>0) if t=+1,  (y<0) if t=-1
   accuracy = classification accuracy (value = double between 0 and 100)
```

----------

#### [dataset.mat](https://github.com/kostasdiamantaras/slackmin/blob/master/dataset.mat)

Matlab file containing the following matrices
```
   x = [18x4500] matrix with the input patterns (4500 patterns of 18 dimensions)
   t = [1x4500] vector with targets (-1/+1)
```

----------

#### [example_exper.m](https://github.com/kostasdiamantaras/slackmin/blob/master/example_exper.m)

Matlab script demonstrating the use and performance of the Slackmin algorithm. It runs a 10-fold cross-validation classification experiment on the data found in *dataset.mat*. Compares the results of Slackmin against [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/).

----------
