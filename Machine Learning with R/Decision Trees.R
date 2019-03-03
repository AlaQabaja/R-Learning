install.packages("C50")
library(C50)
setwd("C:\\Development\\Machine Learning with R\\datasets")
credit = read.csv("credit.csv")
credit$default <- as.factor(credit$default)
set.seed(123)
mytrainSample <- sample(1000, 900)
credit_train <- credit[mytrainSample, ]
credit_test <- credit[-mytrainSample,]
credit_model <- C5.0(x = credit_train[-17], credit_train$default, trials = 10)
credit_pred <- predict(credit_model, credit_test[-17])
CrossTable(credit_test$default, credit_pred, prop.c = FALSE, prop.r = FALSE, prop.chisq = FALSE, dnn = c("Actual","Predicted"))

## Define what error is more costly than the other 
matrix_dimensions <- list(c("no","yes"), c("no","yes"))
names(matrix_dimensions) <- c("predicted","actual")

## We apply an error cost of 4 on false negatives whereas we apply a cost of 1 on false positive 
error_cost <- matrix(c(0,1,4,0), nrow = 2, dimnames = matrix_dimensions)

credit_cost <- C5.0(x = credit_train[-17], credit_train$default, costs = error_cost)
## compared to the boosted version, this model should have more false positives and less false negatives

