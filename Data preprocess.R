## 使用简单的填补方式，向上填充法。（还可以采用rf填补缺失值的方法）
## 对于issueDate的处理采用两种方式，一种是与特定时间间的天数间隔.
##                                  一种是提取月份. 
## 在实际建模效果种，删除近零方差的变量会导致效果下降，因此最终只删除常数变量“policyCode”.
## 在建模之前，把所有的数据类型numeric化，否则会错误识别为character.

path <- "E:/TianChi/Financial/"
loan <- read_csv(paste0(path, "train.csv"))
test <- read_csv(paste0(path, "testA.csv"))

# combine the x_train and x_test
names(select(loan, -isDefault)) == names(test)
comb <- rbind(select(loan, -isDefault), test)

# summary of data:loan
sum_columns <- summarizeColumns(loan)
sum_levels  <- summarizeLevels(loan) #character & logicals will be treated as factors

# divide the numeric variables and character variables
nmr_features <- sum_columns[sum_columns$type == "numeric", ]$name
chr_features <- sum_columns[sum_columns$type == "character", ]$name
nmr_features <- setdiff(nmr_features, "isDefault") 

# divide the continuous variables and categorical variables
cont_cate_in_nmr <- dis_cate_cont(loan, nmr_features)
cate_fea  <- c(cont_cate_in_nmr[[1]], chr_features)
cont_fea  <- cont_cate_in_nmr[[2]]

# missing data
na_columns <- data.frame(colname = sum_columns$name[sum_columns$na != 0], 
                         type = sum_columns$type[sum_columns$na != 0],
                         na =  sum_columns$na[sum_columns$na != 0])


################################################################################## 
################################# data preprocess ################################
##################################################################################

# time data manipulation
# issueDate -- type: Date
# Method 1：the time difference in days
issueDateDT <- as.numeric(comb[["issueDate"]] - as.Date("2007-07-01"))
comb <- comb  %>% mutate(issueDateDT = issueDateDT) %>% select(-issueDate)
# Method 2：extract the month that loans disburse
# comb <- comb %>% mutate(issueDateMon = as.factor(month(comb$issueDate))) %>% select(-issueDate)

# ealiesCreditLine -- type: character
# extract year 
earliesCreditLineYear <- numeric(nrow(comb))
for (i in 1:nrow(comb)) {
  earliesCreditLineYear[i] <- unlist(strsplit(comb$earliesCreditLine[i], split = "-"))[2]
}

comb$CreditLineYear <- as.double(year(today())) - as.double(earliesCreditLineYear)
comb <- comb %>% select(-earliesCreditLine)


## # recode the levels of character variables (subgrade will be turned to dummies)
comb$employmentLength <- as.factor(comb$employmentLength)
comb$employmentLength <- fct_recode(comb$employmentLength, 
                                      `1` = "< 1 year", `2` = "1 year",
                                      `3` = "2 years", `4` = "3 years",
                                      `5` = "4 years", `6` = "5 years",
                                      `7` = "6 years", `8` = "7 years",
                                      `9` = "8 years", `10`= "9 years",
                                      `11`= "10+ years")

comb$grade <- fct_recode(comb$grade, `1` = 'A', `2` = 'B', `3` = 'C',
                           `4` = 'D', `5` = 'E', `6` = 'F', `7` = 'G')

# paste(1:35, names(table(comb$subGrade)), sep = " = ", collapse = ", ")
comb$subGrade <- fct_recode(comb$subGrade, `1` = "A1", `2` = "A2", `3` = "A3", `4` = "A4", `5` = "A5", 
                              `6` = "B1", `7` = "B2", `8` = 'B3', `9` = "B4", `10` = "B5", 
                              `11` = "C1", `12` = "C2", `13` = "C3", `14` = "C4", `15` = "C5", 
                              `16` = "D1", `17` = "D2", `18` = "D3", `19` = "D4", `20` = "D5", 
                              `21` = "E1", `22` = "E2", `23` = "E3", `24` = "E4", `25` = "E5", 
                              `26` = "F1", `27` = "F2", `28` = "F3", `29` = "F4", `30` = "F5", 
                              `31` = "G1", `32` = "G2", `33` = "G3", `34` = "G4", `35` = "G5")
                              
                              
## delete the variables with near zero variance
rm_col <- nearZeroVar(comb, freqCut = 95/5, uniqueCut = 10, saveMetrics = TRUE, 
                      names = TRUE, allowParallel = TRUE)
rm_fea <- rownames(rm_col)[rm_col$nzv == TRUE]
sapply(rm_fea, function(fea) table(comb[[fea]]))
comb_rm_nzv <- comb %>% select(-all_of(rm_fea))

rm_col_extreme <- nearZeroVar(comb, freqCut = 99/1, uniqueCut = 10, saveMetrics = TRUE,
                              name = TRUE, allowParallel = TRUE)
rm_fea_extreme <- rownames(rm_col_extreme)[rm_col_extreme$nzv == TRUE]
sapply(rm_fea_extreme, function(fea) table(comb[[fea]]))
comb_rm_zv <- comb %>% select(-all_of(rm_fea_extreme)) 

for (fea in c(rm_fea, "n13")) {comb[[fea]] <- as.factor(comb[[fea]])}
comb   <- comb %>% select(-policyCode) 
comb$n11 <- fct_collapse(comb$n11, `1` = c("1", "2", "3", "4"))
comb$n12 <- fct_collapse(comb$n12, `1` = c("1", "2", "3", "4"))
comb$n13 <- fct_collapse(comb$n13, `2` = names(table(comb$n13))[-c(1:2)])

# Label-encoding: one-shot, the categorical vars with levels less than 6 and larger than 2.
comb_encode <- comb
onehot_col <- c("homeOwnership", "verificationStatus", "purpose")
for (fea in onehot_col) {comb_encode[[fea]] <- as.factor(comb_encode[[fea]])}
form  <- paste("~ ", paste(onehot_col, collapse = " + "))
dummy <- dummyVars(formula = as.formula(form), data = comb_encode, sep = ".")
pred  <- predict(dummy, newdata = comb_encode)

comb_encode <- cbind(comb_encode %>% select(-all_of(onehot_col)), pred) %>% select(-id)


# missing data (upward)
summ   <- summarizeColumns(comb_encode)
na_col <- summ$name[summ$na != 0]
na_columns  <- sum_columns$name[sum_columns$na != 0]

# setnafill(comb_encode[, na_columns], type = "locf")
# sum(is.na(comb_encode))
comb_imputed <- as_tibble(apply(comb_encode, 2, as.numeric))
for (f in na_columns) {
  if (f == "employmentLength") {
    print(f)
    setnafill(comb_imputed, type = "locf", cols = f)
  }
  if (f %in% cate_fea) {
    print(paste("cate:", f))
    setnafill(comb_imputed, type = "const", fill = getmode(comb_imputed[[f]]), cols = f)
  }
  if (f %in% cont_fea) {
    print(paste("cont:", f))
    setnafill(comb_imputed, type = "const", fill = median(comb_imputed[[f]], na.rm = TRUE), cols = f)
  }
}

sum(is.na(comb_imputed))
ggplot(data = comb_imputed) + 
  geom_bar(mapping = aes(x = employmentLength)) 

par(mfrow = c(2, 1))
ggplot(data = comb_imputed) + geom_density(mapping = aes(x = dti)) + labs(title = "After imputation")
ggplot(data = comb_encode) + geom_density(mapping = aes(x = dti)) + ggtitle("Before imputation")


# mice() (效果不太好)
# imp <- mice(comb_encode, m = 5, method = "midastouch")
# comb_imputed <- complete(imp, action = 3)
# sum(is.na(comb_imputed))


############################################################################## 
###########################  feature selection  ##############################
##############################################################################
'''
issueDateDT
annualIncome
employmentTitle
loanAmnt
term
dti
revolBal
n2
subGrade
ficoRangeLow
n14
installment
homeOwnership.0
homeOwnership.1
interestRate
CreditLineYear
totalAcc
purpose.1
grade
regionCode
n9
n6
revolUtil
n8
title
employmentLength
delinquency_2years
verificationStatus.0
postCode
applicationType
n1
verificationStatus.2
n5
'''
feature_selected <- scan(file = "E:/TianChi/Financial/PythonCode/lgb_feature_chose.txt", what = list(""))[[1]][-c(1:4)]
feature_droped   <- setdiff(names(comb_imputed), feature_selected)
# c('ficoRangeHigh', 'openAcc', 'pubRec', 'pubRecBankruptcies', 'initialListStatus', 'n0',
# 'n3', 'n4', 'n7', 'n10', 'n11', 'n12', 'n13', 'homeOwnership.2', 'homeOwnership.3', 'homeOwnership.4',
# 'homeOwnership.5', 'verificationStatus.1', 'purpose.0', 'purpose.2', 'purpose.3', 'purpose.4', 'purpose.5',
# 'purpose.6', 'purpose.7', 'purpose.8', 'purpose.9', 'purpose.10', 'purpose.11', 'purpose.12', 'purpose.13')

comb_fea_selection <- comb_imputed %>% select(-all_of(feature_droped[1:13]))
save(comb_fea_selection, file = "H:/comb_fea_selection.RData")

################################################################################## 
################################### build model ##################################
##################################################################################
x_train <- comb_imputed[1:800000, ]
x_test <- comb_imputed[800001:1000000, ]

set.seed(666)
trainidx <- createDataPartition(loan$isDefault, p = 0.8, list = FALSE)

x.train <- x_train[trainidx, ]
x.valid <- x_train[-trainidx, ]
y.train <- as.factor(paste0("c", loan$isDefault))[trainidx]
y.valid <- as.factor(paste0("c", loan$isDefault))[-trainidx]

params <- expand.grid(nrounds = 1000, max_depth = 5, eta = 0.05, 
                      subsample = 0.75, colsample_bytree = 0.8, 
                      gamma = 0.2, min_child_weight = 2)
model  <- caret::train(x = x.train, y = y.train, method = "xgbTree", 
                       preProcess = c("center", "scale"),
                       trControl = trainControl("none", classProbs = TRUE),
                       tuneGrid = params)

pred_label_simple <- predict(model, newdata = as.data.frame(x.valid), type = "prob")[, 2]
roc_valid <- roc(as.numeric(valid.loan$isDefault), as.numeric(pred_label_simple))$auc


## cv
xgbControl <- trainControl(method = "cv", number = 5, selectionFunction = "best")
xgbGrid    <- expand.grid(nrounds = 1000, max_depth = 8, eta = 0.05, gamma = 0.6,
                          colsample_bytree = 0.8, min_child_weight = 1.5, subsample = 0.75)

model_tune <- caret::train(x = x.train, y = y.train, method = "xgbTree", 
                           preProcess = c("center", "scale"),
                           trControl = xgbControl, tuneGrid = xgbGrid)

pred_label_tune <- predict(model_tune, newdata = as.data.frame(x.valid), type = "prob")[, 2]

roc_valid <- roc(y.valid, as.numeric(pred_label_tune))
roc_valid$auc


pred_label_test <- predict(model, newdata = as.data.frame(x_test), type = "prob")[, 2]
test_res <- data.frame(id = 800001:1000000, isDefault = round(pred_label_test, 1))
write.csv(x = test_res, file = "E:/TianChi/Financial/test_res.csv", row.names = FALSE)
