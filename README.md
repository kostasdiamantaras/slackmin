# Slackmin Classification Algorithm with example.

This is a MATLAB implementation of the plain vanilla Slackmin algorithm presented in the paper

[M. Kotti, K. I. Diamantaras, "Efficient Binary Classification Through Energy Minimisation of Slack Variables",
Neurocomputing, Elsevier, Vol. 148, pp. 498–511, DOI: 10.1016/j.neucom.2014.07.013, January 2015] (http://www.sciencedirect.com/science/article/pii/S0925231214008911)

This implementation does not contain the search procedures for the finding the optimal hyper-parameters.

### Files:

----------

#### slackmin_train.m
Slackmin training algorithm.

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

#### slackmin_sim.m
Slackmin recall after training

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

#### dataset.mat
Matlab file containing the following matrices
```
   x = [18x4500] matrix with the input patterns
   t = [1x4500] vector with targets (-1/+1)
```
----------
