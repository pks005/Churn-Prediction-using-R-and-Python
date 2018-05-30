rm(list = ls())

library(MLmetrics)
library(corrgram)

train = read.csv("F:\\Work\\Edwisor\\New Site\\6. Projects\\2. Churn Reduction\\Dataset\\Train_data.csv")
test = read.csv("F:\\Work\\Edwisor\\New Site\\6. Projects\\2. Churn Reduction\\Dataset\\Test_data.csv")


# ==============================
# COMBINING TRAIN AND TEST DATA
# ==============================
train$training_data = 1
test$training_data = 2
train_test = rbind(train, test)
train_test_backup = train_test


lapply(train_test, FUN = function(x) length(unique(x)))
lapply(train_test, FUN = function(x) sum(is.na(x)))
lapply(train_test, FUN = function(x) sum(is.null(x)))


# Pre-processing and data cleaning
# ---------------------------------
train_test$Churn = factor(x = train_test$Churn, labels = 0:(length(unique(train_test$Churn))-1))
train_test$voice.mail.plan = factor(x = train_test$voice.mail.plan, labels = 1:length(unique(train_test$voice.mail.plan)))
train_test$international.plan = factor(x = train_test$international.plan, labels = 1:length(unique(train_test$international.plan)))
train_test$phone.number = NULL


par(mar=c(5,5,5,5))
corrgram(train_test, cex.labels = 1)
train_test$total.day.minutes = NULL
train_test$total.eve.minutes = NULL
train_test$total.night.minutes = NULL
train_test$total.intl.minutes = NULL


# ===================
# FEATURE  SELECTION
# ===================
library(Boruta)
b = Boruta(Churn ~ .-training_data, data = train_test, doTrace=1)
par(mar=c(10.1,4.1,4.1,2.1))
plot(b, las=2)
bb = attStats(b)
bb = bb[bb$decision=='Confirmed', ]
selected_cols = rownames(bb)
selected_cols

train_test = train_test[, c(selected_cols, "training_data")]
train_test$Churn = train_test_backup$Churn
train_test$Churn = factor(x = train_test$Churn, labels = 0:(length(unique(train_test$Churn))-1))



# =================
# OUTLIER ANALYSIS
# =================
numeric_cols = lapply(train_test, FUN = function(x) is.numeric(x))
sum=0
for(item in numeric_cols){
  sum = sum + item
}
print(c("Total numeric cols : ", sum))


par(mar=c(1,1,1,1))
par(mfrow=c(1,(sum-1)))
for (x in colnames(train_test[, c(-3,-10,-11)])){
  total = 0
  if (is.numeric(train_test[,x])==T){
    boxplot(train_test[,x], main=x, range = 5)
    outlier_values = boxplot.stats(train_test[,x], coef = 5)$out
    total = total + length(outlier_values)
    print(x)
    print(c("Total Outliers : ", total))
    print(outlier_values)
    cat("\n\n")
  }
}


# Setting all outlier values to NA
# ---------------------------------
par(mar=c(1,1,1,1))
for (x in colnames(train_test[, c(-3,-10,-11)])){
  if (is.numeric(train_test[,x])==T){
    outlier_values = boxplot.stats(train_test[,x], coef = 5)$out
    if (length(outlier_values) != 0){
      train_test[train_test[, x] %in% outlier_values, ][, x] = NA
      cat(c("NA's in ", x, " : ", sum(is.na(train_test[, x])), "\n"))
    }
  }
}


# Imputting NA values 
# --------------------
library(DMwR)
train_test = knnImputation(data = train_test, k = 5)


# ================
# FEATURE SCALING
# ================

# Normalization
# --------------
par(mfrow=c(2,6))
par(col.lab="red")
par(mar=c(5,2,8,1))
# before normalization
colnames(train_test[,4:9])
for(x in colnames(train_test[,4:9])){
  hist(train_test[, x], main = "Before Normalization", xlab = x, ylim = c(0,1500))
  abline(h = 1000, lty=3)
}
# after normalization
for(x in colnames(train_test[,4:9])){
  train_test[, x] = (train_test[, x] - min(train_test[, x]))/(max(train_test[, x])-min(train_test[, x]))
  hist(train_test[, x], main = "After Normalization", xlab = x, col = "green", ylim = c(0,1500))
  abline(h = 1000, lty=3)
}


# Standardization
# ----------------
par(mfrow=c(4,4))
par(mar=c(5,3,2,1))
for(x in colnames(train_test)){
  if(is.numeric(train_test[, x])){
    qqnorm(train_test[, x], xlab = x, main = x)
    qqline(train_test[, x])
    hist(train_test[, x], xlab = x, main = x)
  }
}
train_test_for_standardization_visualization = train_test
for(x in colnames(train_test[,c(4:6,8)])){
  train_test[, x] = (train_test[, x] - mean(train_test[, x]))/(sd(train_test[, x]))
}
# Visualization of graph after standardization
par(mfrow=c(4,2))
par(mar=c(5,5,1,1)+0.1)
for(x in colnames(train_test[,c(4:6,8)])){
  hist(train_test_for_standardization_visualization[, x], xlab = x, main = "Before Standardization", ylim=c(0,1500), xlim = c(-4,4))
  hist(train_test[, x], xlab = x, main = "After Standardization", col = "green", ylim=c(0,1500), xlim = c(-4,4))
}



# ===================================
# SPLITTING DATA INTO TRAIN AND TEST
# ===================================
train = train_test[train_test$training_data ==1, ]
test = train_test[train_test$training_data ==2, ]
train$training_data = NULL
test$training_data = NULL


# ===============
# MODEL BUILDING
# ===============
accuracy = vector()
model_name = vector()
f1_score = vector()
false_negative = vector()

register_model_with_scores = function(name, acc, f1, fn){
  model_name <<- append(model_name, name)
  accuracy <<- append(accuracy, acc)
  f1_score <<- append(f1_score, f1)
  false_negative <<- append(false_negative, fn)
}
par(mfrow=c(1,1))
par(mar=c(4,4,4,4))



# ---> logistic regression
# -------------------------
name = "logistic regression"
logreg = glm(formula = train$Churn ~ ., family = 'binomial', data = train)
predicted = predict(logreg, newdata = test[, names(test) != "Churn"])
predicted = ifelse(predicted>=0.5, 1, 0)
tab = table(actual = test$Churn, predicted=predicted)
tab
acc = sum(diag(tab))/length(predicted)
f1 = F1_Score(y_true = test$Churn, y_pred = predicted)
register_model_with_scores(name, acc, f1, tab[2])
acc
f1


# ---> Decision tree
# -------------------
name = "Decision tree"
library(rpart)
dt = rpart(train$Churn ~ ., data = train, method = "class")
par(mar=c(0,0,0,0))
plot(dt, uniform=TRUE)
?text(dt,use.n=TRUE, all=TRUE, cex=.8)
predicted = predict(dt, newdata = test[, names(test) != "Churn"], type = "class")
tab = table(actual = test$Churn, predicted=predicted)
tab
acc = sum(diag(tab))/length(predicted)
f1 = F1_Score(y_true = test$Churn, y_pred = predicted)
register_model_with_scores(name, acc, f1, tab[2])
acc
f1


# ---> Naive Bayes
# -----------------
name = "Naive Bayes"
library(e1071)
nb = naiveBayes(train$Churn ~ ., data = train, type="class")
predicted = predict(nb, newdata = test[, names(test) != "Churn"],type = "class")
tab = table(actual = test$Churn, predicted=predicted)
tab
acc = sum(diag(tab))/length(predicted)
f1 = F1_Score(y_true = test$Churn, y_pred = predicted)
register_model_with_scores(name, acc, f1, tab[2])
acc
f1


# ---> Random Forest
# -------------------
name = "Random Forest"
par(mfrow=c(1,1))
par(mar=c(4,4,4,4))
library(randomForest)
rf = randomForest(train$Churn ~ ., data = train, ntree=500)
plot(rf)
predicted = predict(rf, newdata = test[, names(test) != "Churn"], type = "class")
tab = table(actual = test$Churn, predicted=predicted)
tab
acc = sum(diag(tab))/length(predicted)
f1 = F1_Score(y_true = test$Churn, y_pred = predicted)
register_model_with_scores(name, acc, f1, tab[2])
acc
f1


# ---> C5.0
# ----------
name = "C5.0"
library(C50)
dt = C5.0(train$Churn ~ ., data = train, method = "class")
plot(dt)
predicted = predict(dt, newdata = test[, names(test) != "Churn"], type = "class")
tab = table(actual = test$Churn, predicted=predicted)
tab
acc = sum(diag(tab))/length(predicted)
f1 = F1_Score(y_true = test$Churn, y_pred = predicted)
register_model_with_scores(name, acc, f1, tab[2])
acc
f1


# ---> J48
# ---------
name = "J48"
library(RWeka)
dt = J48(train$Churn ~ ., data = train)
summary(dt)
plot(dt)
predicted = predict(dt, newdata = test[, names(test) != "Churn"], type = "class")
tab = table(actual = test$Churn, predicted=predicted)
tab
acc = sum(diag(tab))/length(predicted)
f1 = F1_Score(y_true = test$Churn, y_pred = predicted)
register_model_with_scores(name, acc, f1, tab[2])
acc
f1


# --> XGBoost
# ------------
name = "XGBoost"
library(xgboost)
xgb_model <- xgboost(data = data.matrix(train[, names(test) != "Churn"]),
                     early_stopping_rounds = 50, 
                     verbose = T, 
                     label = data.matrix(train$Churn),
                     eta = 0.0001,
                     max_depth = 6, 
                     nround=1000, 
                     subsample = 0.5,
                     colsample_bytree = 0.5,
                     seed = 1,
                     eval_metric = c("error", "auc"),
                     objective = "binary:logistic",
                     nthread = 3
)
predicted <- predict(xgb_model, data.matrix(test[, names(test) != "Churn"]))
predicted = ifelse(predicted>=0.5, 1, 0)
mod = xgb.dump(model = xgb_model, with_stats = T)
mod[1:10]
# feature importance matrix
names <- dimnames(data.matrix(train[, names(test) != "Churn"]))[[2]]
importance_matrix <- xgb.importance(names, model = xgb_model)
xgb.plot.importance(importance_matrix[1:10,])
# Confussion matrix
tab = table(actual = test$Churn, predicted=predicted)
tab
acc = sum(diag(tab))/length(predicted)
f1 = F1_Score(y_true = test$Churn, y_pred = predicted)
register_model_with_scores(name, acc, f1, tab[2])
acc
f1



# ======================
# MODEL'S SCORE SUMMARY
# ======================
models_df = data.frame(model_name, accuracy, f1_score, false_negative)
models_df



# ===========
# CONCLUSION
# ===========

# From the above summary table we can se that best accuracy, F1 score and False Negative Rate is achieved 
# by Random Forest which is 0.9646071 0.9798978 and 54 respectively.

