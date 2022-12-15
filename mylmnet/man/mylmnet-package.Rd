\name{mylmnet-package}
\alias{mylmnet-package}
\alias{mylmnet}
\docType{package}
\title{
  Coordinate descent for linear regression with elastic-net penalty
}
\description{
  This package contains a function called cd_NormalCC, which can calculate the estimate of coefficents for a linear regression model with elastic-net penalty.
}
\details{
  This section should provide a more detailed overview of how to use the
  package, including the most important functions.
}
\author{
Lewen Dai

Maintainer: Lewen Dai <lewen@umich.edu>
}
\references{
Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. Journal of statistical software, 33(1), 1.
}
\keyword{ package }
\seealso{
  https://github.com/lewen07/mylmnet_r
}
\examples{
  set.seed(615)
  alpha = 0.5
  lambda = 0
  n = 10000
  p = 200
  mu0 = runif(1) * 10
  std0 = runif(1) * 10
  X = matrix(rnorm(n*p,mu0, std0),n,p)
  beta = rnorm(p) * 10
  Y = rnorm(n) + X \%*\% beta + 10
  fit = cd_normalCC(X,Y,alpha = alpha,lambda = lambda)
}
