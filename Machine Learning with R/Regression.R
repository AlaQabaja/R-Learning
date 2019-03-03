# When it comes down to it, linear regression tries to solve for the following equation 
# Y = BX + error 
# B = (Xtranspose * X)-1 * Xtranspose * Y
# We can write our own linear model function as such 
library(psych)
reg <- funciton(x, y)
{
  x = as.matrix(x)
  x = cbind(intercept = 1, x)
  b <- solve(t(x) %*% x) %*% t(x) %*% y
  colnames(b) <- "estimate"
  print(b)
  
}

setwd("C:\\Development\\Machine Learning with R\\datasets")
insurance <- read.csv("insurance.csv", stringsAsFactors = FALSE)
pairs.panels(insurance[c("age","bmi","children","expense")])

# To account for non-linear relationship we might need to use a polynomial form of one of the features 
# This will add another B to estimate the new B value and see the effect of the polynomial term
# Y = a + B1X + B2X(squared)

# Sometimes numeric variable can only be effective after a particular value for example BMI
# BMI may only start to have impact after the value of 30. In this case, we better change it to binary variable
# For example if BMI > 35 then 1 else 0. 

# Decision trees can be used to predict numerical values. It is called regression trees. 
# The criteria to split won't be based on entropy, rather, on Standard Deviation Reduction
