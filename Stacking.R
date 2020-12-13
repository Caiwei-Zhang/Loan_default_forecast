# load package
library(data.table)
library(tidyverse)
library(forcats)
library(lubridate)
library(mlr) 
library(ggplot2)
library(missForest)
library(VIM)
library(foreach)
library(doParallel)
library(caret)
library(pROC)
library(corrplot)
library(scorecard)
library(xgboost)
library(randomForestSRC)
library(gbm)

loan <- fread("E:/TianChi/Financial/train.csv") %>% as_tibble()
comb_rm_nzv <- fread("E:/TianChi/Financial/comb_rm_nzv.csv") %>% as_tibble()

x_train <- comb_rm_nzv[1:800000, ]
test    <- comb_rm_nzv[800001:1000000,]
y_train <- loan$isDefault

set.seed(999)
train_idx <- createDataPartition(y_train, p = 0.8, list = FALSE)

x.train <- x_train[train_idx, ]
x.valid <- x_train[-train_idx, ]
y.train <- y_train[train_idx]
y.valid <- y_train[-train_idx]

first_layer <- function(method, x_train, y_train, x_valid, x_test, fold = 5) {
  
  train_list <- createFolds(y_train, k = fold, returnTrain = TRUE)
  
  oof_train_pred  <- numeric(nrow(x_train))
  valid_pred_summ <- NULL
  test_pred_summ  <- NULL
  
  for (k in 1:fold) {
    
    print(paste("Running fold:", k))
    
    cv_train_idx <- train_list[[k]]
    cv_valid_idx <- setdiff(1:nrow(x_train), cv_train_idx)
    
    cv_x_train <- x_train[cv_train_idx, ]
    cv_x_valid <- x_train[cv_valid_idx, ]
    cv_y_train <- y_train[cv_train_idx]
    cv_y_valid <- y_train[cv_valid_idx]
    
    if (method == "gbm") {
      
      print("train model with gbdt.") 
      model <- gbm.fit(x = as.data.frame(cv_x_train), y = cv_y_train, distribution = "bernoulli",
                       n.trees = 1000, interaction.depth = 6, n.minobsinnode = 20, shrinkage = 0.05, 
                       nTrain = nrow(cv_x_train))
      
      oof_pred_valid <- predict(model, as.data.frame(cv_x_valid), type = "response")
      
      pred_valid <- predict(model, x_valid, type = "response")
      pred_test  <- predict(model, x_test,  type = "response")

    }
    
    if (method == "xgb") {
      
      print("train model with xgboost.")
      
      cv_dtrain <- xgb.DMatrix(data = as.matrix(cv_x_train), label = cv_y_train)
      cv_dvalid <- xgb.DMatrix(data = as.matrix(cv_x_valid), label = cv_y_valid)
      
      dvalid  <- xgb.DMatrix(data = as.matrix(x_valid))
      dtest   <- xgb.DMatrix(data = as.matrix(x_test))
      
      param  <- list(eta = 0.05, gamma = 0.8, max_depth = 6, min_child_weight = 1.5, 
                     subsample = 0.8, colsample_bytree = 0.8, # lambda = 0.5, alpha = 0.5,
                     objective = "binary:logistic", eval_metric = "auc", nthread = 10)
      
      model  <- xgb.train(data = cv_dtrain, nrounds = 5000, params = param,
                          watchlist = list(train = cv_dtrain, eval = cv_dvalid), 
                          early_stopping_rounds = 200, print_every_n = 500)
      
      oof_pred_valid <- to_label(predict(model, newdata = cv_dvalid, ntreelimit = model$best_ntreelimit))
      
      pred_valid <- predict(model, newdata = dvalid, ntreelimit = model$best_ntreelimit)
      pred_test  <- predict(model, newdata = dtest, ntreelimit = model$best_ntreelimit)
      
    }
    
    ### add lightgbm
    
    oof_train_pred[cv_valid_idx] <-  oof_pred_valid
    valid_pred_summ <- cbind(valid_pred_summ, pred_valid)
    test_pred_summ  <- cbind(test_pred_summ, pred_test)
    
  }
  
  res <- list(oof_train_pred, rowMeans(valid_pred_summ), rowMeans(test_pred_summ))
  return (res)
  
}

to_label <- function(x) {
  new_x <- sapply(x, function(i) { if (i >= 0.5) {i = 1} else {i = 0}})
  new_x
}


# method <- c("gbm", "xgb")
# x_train_stack <- x_valid_stack <- x_test_stack <- NULL
# lapply(methods, function(mth) {
#   temp <- first_layer(method = mth, x_train = x.train, y_train = y.train, x_valid = x.valid, x_test = test)
#   x_train_stack <- cbind(x_train_stack, temp[[1]])
#   x_valid_stack <- cbind(x_valid_stack, temp[[2]])
#   x_test_stack  <- cbind(x_test_stack, temp[[3]])})

lgb_train <- fread("E:/TianChi/Financial/PythonCode/lgb_train.csv")
lgb_valid <- fread("E:/TianChi/Financial/PythonCode/lgb_valid.csv")
lgb_test  <- fread("E:/TianChi/Financial/PythonCode/lgb_test.csv") 

lgb_label[[1]] <- lgb_train[[1]]
lgb_label[[2]] <- to_label(lgb_valid[[1]])
lgb_label[[3]] <- to_label(lgb_test[[1]])

x_train_stack <- data.frame(gbm = gbm_res[[1]], xgb = xgb_res[[1]]) #, lgb = lgb_label[[1]])
x_valid_stack <- data.frame(gbm = gbm_res[[2]], xgb = xgb_res[[2]]) #, lgb = lgb_label[[2]])
x_test_stack  <- data.frame(gbm = gbm_res[[3]], xgb = xgb_res[[3]]) #, lgb = lgb_label[[3]])

logistic_model <- glm(formula = label ~., data = data.frame(x_train_stack, label = y.train), family = binomial())

pred_valid <- predict(logistic_model, newdata = x_valid_stack, type = "response") 
auc_valid  <- roc(y.valid, pred_valid)$auc
print(paste("The auc on valid data is", auc_valid))

pred_test  <- predict(logistic_model, newdata = x_test_stack, type = "response")



## drop
############################# rank average ############################
# p <- setwd("H:/")
# file <- list.files(p)
# csv.file <- file[grep(".csv", file)]
# res <- NULL
# for (file in csv.file) {
#   res <- cbind(res, read_csv(file)[[2]])
# }
# 
# res_rank <- apply(res, 2, rank)
# avr_rank <- rowMeans(res_rank)
# 
# # M1:
# z_rank <- (avr_rank - min(avr_rank))/(max(avr_rank)-min(avr_rank))
# test_res_rank_avr <- data.frame(id = 800001:1000000, isDefault = z_rank)
# write.csv(test_res_rank_avr, file = "E:/TianChi/Financial/test_res_rank_avr.csv", row.names = FALSE)
# 
# #M2:
# harmonic <- function(vec){
#   vec_inv <- 1/vec
#   summ <- sum(vec_inv)
#   return(summ)
# }
# 
# harm_avr_rank <- apply(res_rank, 1, harmonic)
# test_res_harm_rank_avr <- data.frame(id = 800001:1000000, isDefault = harm_avr_rank)
# write.csv(test_res_harm_rank_avr, file = "E:/TianChi/Financial/test_res_harm_rank_avr.csv", row.names = FALSE)

