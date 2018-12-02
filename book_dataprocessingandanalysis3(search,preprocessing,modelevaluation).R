# 1. Data Searching ----
# Technical Statistics ----
# describe(x, exclude.missing = T), describe(x, data, na.action) ----
# em : T -> place NA at the end
#      F -> print technical statistics about only saving NA
# summary.formula(formula, data, na.action = NULL, fun = NULL, ----
# method = c("response", "reverse", "cross"))
# fun : NULL -> mean
library(Hmisc)
str(mtcars)
describe(mtcars)
summary(mtcars)
summary(mpg ~ cyl + hp, data = mtcars)
summary(mpg ~ cyl + hp, data = mtcars, fun = var)
# summarising rhs according to category of lhs
summary(cyl ~ mpg + hp, data = mtcars, method = "reverse")
# summarising lhs according to combination of rhs
summary(mpg ~ cyl + hp, data = mtcars, method = "cross")
example("summary.formula")

# Data Visualization ----
# featurePlot(x, y, plot = if(is.factor(y))) ----
library(caret)
plot(iris)
plot(iris$Sepal.Length)
plot(Species ~ Sepal.Length, data = iris)
with(iris, {
  plot(Sepal.Length, Sepal.Width, pch = as.numeric(Species))
  legend("topright", legend = levels(iris$Species), pch = 1 : 3)
})
install.packages("ellipse") # Viewport 'plot_01.panel.1.1.off.vp' was not found
par(ask = F) # turn off "Hit <Return> to see next plot"
featurePlot(iris[, 1 : 4], iris$Species, "ellipse")

# 2. Preprocessing ----
# (1) Data Transforming ----
# Feature Scaling : KNN, SVM, NN etc. ----
# scale(x, center = T, scale = T) ----
# center : T -> subtracting mean of total data from total data
# scale : T & center : T -> dividing total data into sd of total data
# scale : T & center : F -> dividing total data into root mean square of total data
# scale : F -> divide nothing
## RMS(root mean square) : square((x1^2 + x2^2 + ... + xn^2) / n)
cbind(as.data.frame(scale(iris[1 : 4])), iris$Species)
# PCA(Principal Component Analysis) ----
# princomp(x, cor = F) ----
# cor : T -> correlation, F -> covariance
x <- 1 : 10
y <- x + runif(10, min = -.5, max = .5)
z <- x + y + runif(10, min = -10, max = .10)
data <- data.frame(x, y, z); data
pr <- princomp(data)
summary(pr)
pr$scores
# One hot encoding ----
all <- factor(c(paste0(LETTERS, "0"), paste0(LETTERS, "1")))
data <- data.frame(lvl = all, value = rnorm(length(all)))
library(randomForest)
m <- randomForest(value ~ lvl, data = data)
# model.matrix(object, data) ----
x <- data.frame(lvl = factor(c("A", "B", "A", "A", "C")),
                value = c(1, 3, 2, 4, 5)); x
model.matrix(~ lvl, data = x)[, -1]
# NA ----
# complete.cases, is.na, DMwR::centralImputation, DMwR::knnImputation
iris_na <- iris
iris_na[c(10, 20, 25, 40, 32), 3] <- NA
iris_na[c(33, 100, 123), 1] <- NA
iris_na[!complete.cases(iris_na), ]
iris_na[is.na(iris_na$Sepal.Length), ]
mapply(median, iris_na[1 : 4], na.rm = T)
library(DMwR)
iris_na[!complete.cases(iris_na), ]
centralImputation(iris_na[1 : 4])[c(10, 20, 25, 32, 33, 40, 100, 123), ]
knnImputation(iris_na[1 : 4])[c(10, 20, 25, 32, 33, 40, 100, 123), ]
# Variable Selection, Feature Selection ----
# (1) Filter Method : 특정 모델링 기법에 의존하지 않고 데이터의 통계적 특성
#                     (상호 정보량, 상관 계수 ...)으로부터 변수를 택하는 방법
# (2) Wrapper Method : 변수의 일부분만을 모델링에 사용하고 그 결과를 확인하는
#                      작업을 반복하면서 변수를 택해나가는 방법
# (3) Embedded Method : 모델 자체에 변수 선택이 포함된 방법(LASSO)
# Near Zero Variance ----
# caret::nearZeroVar(x, freqCut = 95 / 5, uniqueCut = 10, saveMetrics = F) ----
#                    freqCut - The cutoff for the ratio of 
#                              the number of the most frequently observed values 
#                              to the number of the second most frequently observed values.
#                    uniqueCut - The cutoff for the ratio of
#                                the number of the total values
#                                to the number of the unique values.
library(caret)
library(mlbench)
data(Soybean)
nearZeroVar(Soybean, saveMetrics = T)
nearZeroVar(Soybean)
mySoybean <- Soybean[, -nearZeroVar(Soybean)]
# Correlation ----
# High correlation
# https://stackoverflow.com/questions/14813884/correlated-features-and-classification-accuracy
# caret::findCorrelation(x, cutoff = .90), FSelector::linear.correlation
# FSelector::rank.correlation, FSelector::cutoff.k(attrs, k), cutoff.k.percent(attrs, k)
# x : matrix of correlation
library(mlbench)
library(caret)
data(Vehicle)
findCorrelation(cor(subset(Vehicle, select = -c(Class))))
cor(subset(Vehicle, select = -c(Class)))[c(3, 8, 11, 7, 9, 2), c(3, 8, 11, 7, 9, 2)]
myVehicle <- Vehicle[, -c(3, 8, 11, 7, 9, 2)]
install.packages("rJava")
library(rJava)
install.packages("FSelector")
library(FSelector)
library(mlbench)
data(Ozone)
v <- linear.correlation(V4 ~ ., data = subset(Ozone, select = -c(V1, V2, V3)))
cutoff.k(v, 3)
# Chi-Squared Test ----
# FSelector::chi.squared(formula, data) ----
data(Vehicle)
cs <- chi.squared(Class ~ ., data = Vehicle); cs
cutoff.k(cs, 3)
# Importance of variable from model ----
# caret::varImp(object) : 의사결정나무에서 가지가 나뉠 때의 ----
#                         손실 함수 감소량을 각 변수에 더하는 방법으로 변수 중요도 평가
library(mlbench)
library(rpart)
library(caret)
data("BreastCancer")
m <- rpart(Class ~ ., data = BreastCancer)
varImp(m)

# 3. Model Evaluating Method
# Confusion Matrix ----
# 실제 / 예측 Y : True Positive(TP)
# 실제 N / 예측 Y : False Positive(FP)
# 실제 Y / 예측 N : False Negative(FN)
# 실제 N / 예측 N : True Negative(TN)
# Precision : TP / (TP + FP) - Y로 예측된 것 중 실제로도 Y인 경우
# Accuracy : (TP + FN) / (TP + FP + FN + TN) - 전체 예측에서 옳은 예측
# Recall(Sensitivity, TP Rate, Hit Rate) : TP / (TP + FN) - 실제로 Y인 것들 중 예측이 Y로 된 경우
# Specificity : TN / (FP + TN) - 실제로 N인 것들 중 예측이 N으로 된 경우의 비율
# FP Rate : FP / (FP + TN) - 실제로 N인 것들 중 Y로 예측된 경우. (1 - Specificity)
# F1 : 2 * Precision * Recall / (Precision + Recall) - Precision과 Recall의 조화 평균
# Kappa : (Accuracy - P(e)) / (1 - P(e)) - 두 평가자의 평가가 얼마나 일치하는지 평가하는 값
#                                          P(e) -> 두 평가자의 평가가 우연히 일치할 확률
#                                       {(TP + FP) * (TP + FN) * (FN + TN) * (FP + TN)} / Total^2
predicted <- factor(c(1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1),
                    levels = c(0, 1))
actual <- factor(c(1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1),
                 levels = c(0, 1))
xtabs(~ predicted + actual)
sum(predicted == actual) / NROW(actual)
# e1071::confusionMatrix(data, reference)
install.packages("e1071")
library(e1071)
confusionMatrix(predicted, actual) # No Information Rate : 가장 많은 값이 발견된 분류의 비율

# ROC Curve ----
# ROCR::prediction(predictions, lables) ----
# ROCR::performance(prediction.obj, measure, x.measure) ----
# measure - acc(Accuracy), fpr(FP Rate), tpr(TP Rate), rec(Recall)
set.seed(137)
probs <- runif(100)
labels <- as.factor(ifelse(probs > .5 & runif(100) < .4, "A", "B"))
install.packages("ROCR")
library(ROCR)
pred <- prediction(probs, labels); pred
plot(performance(pred, "tpr", "fpr"))
plot(performance(pred, "acc", "cutoff"))
performance(pred, "auc")

# Cross Validation ----
# cvTools::cvFolds(n, K = 5, R = 1, type = c("Random", "consecutive", "interleaved")) ----
install.packages("cvTools")
library(cvTools)
cvFolds(10, K = 5, type = "random")
cvFolds(10, K = 5, type = "consecutive")
cvFolds(10, K = 5, type = "interleaved")
set.seed(719)
cv <- cvFolds(NROW(iris), K = 10, R = 3); cv
head(cv$which, 20)
head(cv$subset)
validation_idx <- cv$subset[which(cv$which == 1), 1]; validation_idx
train <- iris[-validation_idx, ]
validation <- iris[validation_idx, ]
# caret::createDataPartition(y, times = 1, p = 0.5, list = T) ----
# y - Category(or Label), times - Number of Division
# p - ratio of data that will use at train data
# caret::createFolds(y, k = 10, list = T, returnTrain = F) ----
# caret::createMultiFolds(y, k = 10, times = 5) ----
library(caret)
parts <- createDataPartition(iris$Species, p = 0.8); parts
table(iris[parts$Resample1, "Species"])
table(iris[-parts$Resample1, "Species"])
createFolds(iris$Species, k = 10)
createMultiFolds(iris$Species, k = 10, times = 3)
