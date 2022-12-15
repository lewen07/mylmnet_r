# mylmnet_r
This is a package reimplements the coordinate descent algorithm for linear regression with elastic-net penalty. It contains one function cd_normalCC which takes features matrix $X \in R^{N \times p}$, reponse variable vector $Y \in R^N$, parameter $\alpha \in [0,1]$ controlling compensation between ridge and lasso, vector $\lambda \geq 0$ balancing between model precision and simplicity, parametr maxit = 10000 setting the maximum number of iteration, parameter tol= 1e-7 controlling convergence and parameter standardize = TRUE controlling if standardize each column of $X$.

Here is the example testing code
```{r}
set.seed(615)
alpha = 0.5
lambda = 0
n = 10000
p = 200
mu0 = runif(1) * 10
std0 = runif(1) * 10
X = matrix(rnorm(n*p,mu0, std0),n,p)
beta = rnorm(p) * 10
Y = rnorm(n) + X%*% beta + 10
fit = cd_normalCC(X,Y,alpha = alpha,lambda = lambda)
```
