//
//  main.cpp
//  final
//
//  Created by 莺时 on 11/28/22.
//

#include <iostream>
#include <math.h>


#include <Rcpp.h>

using namespace Rcpp;


Rcpp::NumericVector meanC(Rcpp::NumericMatrix x){
  int nrow = x.nrow(), ncol = x.ncol();
  Rcpp::NumericVector out(ncol);
  for (int i = 0; i < ncol; i++) {
    double total = 0;
    for (int j = 0; j < nrow; j++) {
      total += x(j, i);
    }
    out[i] = total;
  }
  return out/nrow;
}


Rcpp::NumericVector stdC(Rcpp::NumericMatrix x){
  int nrow = x.nrow(), ncol = x.ncol();
  Rcpp::NumericVector m = meanC(x);
  Rcpp:: NumericVector out(ncol);
  Rcpp::NumericMatrix a(nrow,ncol);
  for (int i = 0; i < ncol; i++) {
    double total = 0;
    for (int j = 0; j < nrow; j++) {
      a(j,i) = x(j,i) - m[i];
      a(j,i) *= a(j,i);
      total += a(j, i);
    }
    out[i] = total;
  }
  return sqrt(out/(nrow-1));
}


Rcpp::NumericMatrix standard_C(Rcpp::NumericMatrix x){
  int nrow = x.nrow(), ncol = x.ncol();
  Rcpp::NumericVector m = meanC(x);
  Rcpp::NumericVector sd = stdC(x);
  Rcpp::NumericMatrix out(nrow,ncol);
  for (int i = 0; i < ncol; i++) {
    for (int j = 0; j < nrow; j++) {
      out(j,i) = (x(j,i) - m[i])/sd[i];
    }
  }
  return out;
}


double soft_threshC(double x, double gamma){
  double relu_inner = abs(x) - gamma;
  double out;
  int flag = (x > 0) - (x < 0);
  if (relu_inner > 0){
    out = relu_inner;
  } else{
    out = 0;
  }
  return flag * out;
}


double vv(NumericVector x, NumericVector res){
  double a = 0;
  int n = x.length();
  for (int i = 0; i < n; i++){
    a += x[i] * res[i];
  }
  return a;
}


NumericVector sweep(NumericVector beta, NumericVector std){
  int p = beta.length();
  NumericVector beta_r(p);
  for(int i = 0; i < p; i++){
    beta_r(i) = double(beta(i))/std(i);
  }
  return beta_r;
}


NumericVector va(NumericVector beta, NumericVector mat_j){
  int p = beta.length();
  NumericVector out(p);
  for(int i = 0; i < p; i++){
    out(i) = beta(i) + mat_j(i);
  }
  return out;
}




 List cd_updateC(NumericMatrix x, NumericVector y, double alpha, double lambda, int maxit, double tol,NumericVector mu, NumericVector std){
  int convergence = 1;
  int n = x.nrow(),  p = x.ncol();
  int iter = 0;
  NumericVector beta(p);
  NumericVector res = clone(y);
  double intercept;
  //res = res - mean(res);
  double obj0 = 1.0/(2 * n) * vv(res,res) + lambda * ( 0.5 *(1.0 - alpha) * vv(beta,beta) + alpha * sum(abs(beta)));
  while(iter < maxit){
    iter ++;
    for(int j = 0; j < p; j ++){
      double beta0 = beta[j];
      double relu_inner = 1.0/n * vv(x(_,j),res) + beta(j);
      double update_st = soft_threshC(relu_inner, lambda * alpha);
      beta(j) = update_st/(1 + lambda * (1.0 - alpha));
      res = res + x(_,j) * ( - beta(j) + beta0);
    }
    double obj = 1.0/(2 * n) * vv(res,res) + lambda * ( 0.5 *(1 - alpha) * vv(beta,beta) + alpha * sum(abs(beta)));
    double delta = abs(obj0 - obj);
    double accuracy = (abs(obj0) + abs(obj))*tol;
    if(delta < (accuracy + tol)){
      convergence = 0; // converged
      //obj0 = obj;
      beta = sweep(beta, std);
      intercept = mean(y) - vv(beta,mu);
      break;
    }
    obj0 = obj;
  }
  return List::create(Named("beta") = beta,
                      Named("iter")       =  iter,
                      Named("convergence") = convergence,
                      Named("intercept") = intercept);
} 



// [[Rcpp::export]]
List cd_normalCC(NumericMatrix X, NumericVector y, double alpha, NumericVector lambda_vec, int maxit = 10000, double tol = 1e-7, bool standardize = true){
  int p = X.ncol();
  NumericMatrix x = X;
  NumericVector mu = meanC(x);
  NumericVector std = stdC(x);
  if(standardize){
    //std::cout << X(0,0);
    x = standard_C(X);
  }
  //std::cout << X(0,0);
  int T = lambda_vec.length();
  NumericMatrix beta_mat(p,T);
  NumericVector con_vec(T);
  NumericVector iter_vec(T);
  NumericVector int_vec(T);
  for(int t = 0; t < T; t++){
    //std::cout << y(0);
    List out = cd_updateC(x, y, alpha, lambda_vec(t), maxit, tol,mu, std);
    //std::cout << x(0,0);
    beta_mat(_,t) = va(out["beta"],beta_mat(_,t));
    //std::cout << y(0);
    con_vec(t) = out["convergence"];
    iter_vec(t) = out["iter"];
    int_vec(t) = out["intercept"];
  }
  return List::create(Named("beta") = beta_mat,
                      Named("iter")       =  iter_vec,
                      Named("convergence") = con_vec,
                      Named("intercept") = int_vec);
}








