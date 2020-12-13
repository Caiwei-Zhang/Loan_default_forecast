# get the mode of a vector
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}


# distinguish the continuous variables and categorical variables 
dis_cont_cate <- function(data, feas) {
  
  numerical_noserial_fea = vector("character", 0)
  numerical_serial_fea = vector("character", 0)
  
  for (fea in feas) {
    temp = length(unique(data[[fea]]))
    if (temp <= 30) {
      numerical_noserial_fea <rial_fea(numerical_noserial_fea, fea)
    } else {
      numerical_serial_fea <- append(numerical_serial_fea, fea)
    }
  }
  
  return (list(numerical_noserial_fea, numerical_serial_fea))
  
}



# cross validation by xgboost model
cv_model <- function(data, x.train, label, x_test, fold, caret = FALSE) {
  
  if(missing(data)){
    train_list <- createFolds(label, k = fold, returnTrain = TRUE)
    X <- x.train
    y <- label
  } else {
    train_list <- createFolds(data[[label]], k = fold, returnTrain = TRUE)
    X <- select(data, -all_of(label))
    y <- data[[label]]
  } 
  
  cv_score   <- NULL
  test_res   <- NULL
  
  for (k in 1:fold) {
    
    sprintf("******************************** [%g] *********************************", k)
    
    train_idx <- train_list[[k]]

    x_train <- X[train_idx, ]
    x_valid <- X[-train_idx, ]
    y_train <- y[train_idx]
    y_valid <- y[-train_idx]
    
    # build a model with caret
    if (caret == TRUE) {
      
      xgbGrid <- expand.grid(nrounds = 1000, max_depth = 5, eta = 0.05, 
                             subsample = 0.75, colsample_bytree = 0.8, 
                             gamma = 0.2, min_child_weight = 2)
      
      model  <- caret::train(x = x_train, y = y_train, method = "xgbTree", 
                             preProcess = c("center", "scale"),
                             trControl = trainControl("none", classProbs = TRUE),
                             tuneGrid = params)    
      
      pred_valid <- predict(model, newdata = as.data.frame(x_valid), type = "prob")[, 2]
      auc_valid <- roc(y_valid, pred_valid)$auc
      
    } else {
      
      dtrain <- xgb.DMatrix(data = as.matrix(x_train), label = y_train)
      dvalid <- xgb.DMatrix(data = as.matrix(x_valid), label = y_valid)
      dtest  <- xgb.DMatrix(data = as.matrix(x_test))
      
      param  <- list(eta = 0.05, gamma = 0.8, max_depth = 5, min_child_weight = 1.5, 
                     subsample = 0.7, colsample_bytree = 0.7, # lambda = 0.5, alpha = 0.5,
                     objective = "binary:logistic", eval_metric = "auc", nthread = 10)
      watchlist <- list(train = dtrain, eval = dvalid)
      model     <- xgb.train(data = dtrain, nrounds = 5000, watchlist = watchlist, 
                             early_stopping_rounds = 200, verbose = 1, 
                             print_every_n = 200, params = param)
      
      pred_valid <- predict(model, newdata = dvalid, ntreelimit = model$best_ntreelimit)
      auc_valid  <- roc(y_valid, pred_valid)$auc
      
      pred_test  <- predict(model, newdata = dtest, ntreelimit = model$best_ntreelimit)
    }
    
    cv_score <- append(cv_score, auc_valid)
    test_res <- cbind(test_res, pred_test)
    sprintf("The auc of %g th fold is %f", k, cv_score[k])
    
  }
  
  print(cv_score)
  print(paste("score mean:", mean(cv_score)))
  print(paste("score standard deviance:", sd(cv_score)))
  
  return (test_res)
}

check_ga0.8 <- cv_model(x.train = x_train, label = loan$isDefault, x_test = x_test, fold = 5)
test_res_check_ga0.8 <- data.frame(id = 800001:1000000, isDefault = rowMeans(check_ga0.8))

write.csv(test_res_check_ga0.8, file = "E:/TianChi/Financial/test_res_check_ga0.8.csv", row.names = FALSE)





## Grid search
xgbGrid <- expand.grid(nrounds = 2000, eta = 0.1, gamma = 0.6, max_depth = seq(4, 9, 1), 
                       min_child_weight = 1.5, subsample = 0.7, colsample_bytree = 0.7, # lambda = 0.5, alpha = 0.5,
                       objective = "binary:logistic", eval_metric = c("rmse", "auc"), 
                       nthread = 15)
# xgbGrid <- xgbGrid[!duplicated(xgbGrid), ]

# data type preparation
X <- select(loan, -isDefault)
y <- data[[isDefault]]

Ddata <- xgb.DMatrix(data = as.matrix(X), label = y)

cv_tune <- function(param, nrounds = 2000, nfold = 5, early_stopping_rounds = 200) {
  
  # tune parameters by xgb.cv()
  cv.model  <- caret::train(data = loan, method = "xgbTree", nfold = nfold, verbose = 1,
                      early_stopping_rounds = early_stopping_rounds, print_every_n = 200, 
                      trControl = trainControl("none", classProbs = TRUE), tuneGrid = xgbGrid)
  print(cv.model$bestTune)
  print(cv.model$metric)
  
  return (cv.model)
}

  
