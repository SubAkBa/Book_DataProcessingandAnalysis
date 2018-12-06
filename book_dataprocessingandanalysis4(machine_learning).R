# 1. Logistic Regression - log(p / (1 - p)) = B0 + B1 * X ----
# glm(formula, data, family) ----
# predict.glm(object, newdata, type = c("link", "response", "terms")) ----
# link - 선형 독립 변수들의 연산 결과의 크기로 값을 반환 (log(p / (1 - p)))
# response - 반응 변수의 크기로 값을 반환 (p)
# terms - 행렬에 모델 포뮬러의 각 변수에 대한 적합된 값을 선형 예측 변수의 크기로 반환
d <- subset(iris, Species == "virginica" | Species == "versicolor")
str(d)
d$Species <- factor(d$Species)
str(d)
m <- glm(Species ~ ., data = d, family = "binomial"); m
fitted(m)[c(1 : 5, 51 : 55)]
f <- fitted(m)
as.numeric(d$Species)
is_correct <- ifelse(f > .5, 1, 0) == as.numeric(d$Species) - 1
sum(is_correct)
sum(is_correct) / NROW(is_correct)
predict(m, newdata = d[c(1, 10, 55), ], type = "response")


# 2. Multinomial Logistic Regression ----
# nnet::multinom(formula, data), fitted(object) ----
## Machine Learning Model is created by modifying parameters to better perform classification 
## predict on training data. this is called that the model is fitted on training data.
# predict.multinom(object, newdata, type = c("class", "probs"))
# class - Category, probs - Probability of each classification
library(nnet)
m <- multinom(Species ~ ., data = iris); m
head(fitted(m))
predict(m, newdata = iris[c(1, 51, 101), ], type = "class")
predict(m, newdata = iris, type = "probs")
predicted <- predict(m, newdata = iris)
sum(predicted == iris$Species) / NROW(predicted)
xtabs(~ predicted + iris$Species)


# 3. Decision Tree ----
# CART : Classification and Regression Trees
# rpart::rpart(formula, data) : create Recursive Partitioning and Regression Trees ----
# rpart::predict.rpart(object, newdata, type = c("vector", "prob", "class", "matrix")) ----
# rpart::plot.rpart(x, uniform = FALSE, branch = 1, compress = FALSE, nspace, margin = 1) ----
# uniform - T : 노드 간의 간격이 일정하게 그려진다. F : 노드마다 에러에 비례한 간격이 주어진다.
# branch : 가지 모양 설정, 0 - V Shape, 1 - Square
# compress - T : 나무를 좀 더 빽빽하게 그린다.
# nspace : 노드와 자식 노드 간의 간격, margin : 그림주변에 위치한 잎사귀 노드에 추가로 부여할 여백
# rpart.plot::prp(x, type = 0, extra = 0, digits = 2)
install.packages("rpart")
library(rpart)
(m <- rpart(Species ~ ., data = iris))
plot(m, compress = T, margin = .2)
text(m, cex = 1.5)
install.packages("rpart.plot")
library(rpart.plot)
prp(m, type = 4, extra = 2, digits = 3)
head(predict(m, newdata = iris, type = "class"))

# Conditional Inference Tree ----
# party::ctree(formula, data) ----
# party::predict.BinaryTree(obj, newdata, type = c("response", "node", "prob")) ----
# response - 분류, node - 잎사귀 노드 ID, prob - 각 분류에 대한 확률
install.packages("party")
library(party)
(m <- ctree(Species ~ ., data = iris))
plot(m)
levels(iris$Species)

# Random Forest ----
# 1. 데이터의 일부를 복원 추출로 꺼내고 해당 데이터에 대해서만 의사 결정 나무를 만든다.
# 2. 노드내 데이터를 자식 노드로 나누는 기준을 정할 때 전체 변수가 아니라 
#    일부 변수만 대상으로 하여 가지를 나눌 기준을 찾는 방법
# randomForest::randomForest(formula, data, ntree = 500, mtry, importance = F) ----
# ntree - number of tree, mtry - 노드를 나눌 기준을 정할 때 고려할 변수의 수
# importance - whether importance of variable is evaluated
# randomForest::predict(object, newdata, type = c("response", "prob", "vote"))
# response - predicted value, prob - matrix of predicted probability
# vote - matrix of vote result
# randomForest::importance(x, type = NULL) type - 1 : Accuracy, 2 : Impurity ----
# randomForest::varImpPlot(x, type = NULL) ----
install.packages("randomForest")
library(randomForest)
(m <- randomForest(Species ~ ., data = iris))
head(predict(m, newdata = iris))
(m <- randomForest(iris[, 1 : 4], iris[, 5]))

m <- randomForest(Species ~ ., data = iris, importance = T)
importance(m)
varImpPlot(m, main = "varImpPlot of iris")

# expand.grid(...)
(grid <- expand.grid(ntree = c(10, 100, 200), mtry = c(3, 4)))

# Example
library(cvTools)
library(foreach)
library(randomForest)
set.seed(719)
K = 10; R = 3
cv <- cvFolds(NROW(iris), K = K, R = R)

grid <- expand.grid(ntree = c(10, 100, 200), mtry = c(3, 4))

result <- foreach(g = 1 : NROW(grid), .combine = rbind) %do% {
  foreach(r = 1 : R, .combine = rbind) %do% {
    foreach(k = 1 : K, .combine = rbind) %do% {
      validation_idx <- cv$subsets[which(cv$which == k), r]
      train <- iris[-validation_idx, ]
      validation <- iris[validation_idx, ]
      # 모델 훈련
      m <- randomForest(Species ~ .,
                        data = train,
                        ntree = grid[g, "ntree"],
                        mtry = grid[g, "mtry"])
      # 예측
      predicted <- predict(m, newdata = validation)
      # 성능 평가
      precision <- sum(predicted == validation$Species) / NROW(predicted)
      return(data.frame(g = g, precision = precision))
    }
  }
}

library(plyr)
ddply(result, .(g), summarize, mean_precision = mean(precision))
grid[c(1, 3), ]


# 4. Neural Network ----
# nnet::nnet(formula, data, weights, ...) ----
# nnet::nnet(x, y, weights, size, Wts, mask, linout = F, entropy = F, softmax = F, MaxNWts = 1000) ----
## size - number of hidden layer node, Wts - initial weight value
## mask - parameter to optimize(default - all parameters)
## linout - T : y = ax + b 같은 linear output이 활성 함수로 사용
##          F : logistic(sigmoid) function이 활성 함수로 사용
## entropy(모델 학습시 모델의 출력과 원하는 값을 비교할 때 사용할 함수) - T : entropy
##                                                                        F : SSE
## softmax : whether softmax function is used at output layer
## MaxNWts : maximum number of weights
# nnet::predict.nnet(object, newdata, type = c("raw", "class"))
# raw - matrix that neural net returns, class - predicted classification
library(nnet)
(m <- nnet(Species ~ ., data = iris, size = 3))
predict(m, newdata = iris)
predict(m, newdata = iris, type = "class")
# Y의 레벨이 2개라면 entropy 사용하여 parameter 추정, 3개 이상이라면 SSE로 파라미터가 추정되고 softmax 적용
# nnet::class.ind(class)
class.ind(iris$Species)
(m2 <- nnet(iris[, 1 : 4], class.ind(iris$Species), size = 3, softmax = T))
predict(m2, newdata = iris[, 1 : 4], type = "class")


# 5. SVM(Support Vector Machine) ----
# kernlab::ksvm(x, data = NULL) ----
# kernlab::ksvm(x, y = NULL, scaled = T, kernel = "rbfdot", kpar = "automatic") ----
# scaled - T : transforming data that mean 0, variance 1
# kernel - rdfdot : Radial Basis Function (kernlab::dots)
# kpar : 휴리스틱으로 적절한 파라미터를 찾는다.
# kernlab::predict.ksvm(object, newdata, type = "response") ----
# e1071::svm(formula, data = NULL) ----
# e1071::svm(x, y = NULL, scale = T, type = NULL, kernel = "radial", ----
#            gamma = if(is.vector(x)) 1 else 1 / ncol(x), cost = 1)
# e1071::tune(method, train.x, train.y, data, ...) : perform parameter tuning that using grid search ----
## grid search : how to test for all possible cases given as arguments
install.packages("kernlab")
library(kernlab)
(m <- ksvm(Species ~ ., data = iris))
head(predict(m, newdata = iris))
ksvm(Species ~ ., data = iris, kernel = "vanilladot")
(m <- ksvm(Species ~ ., data = iris, kernel = "polydot", kpar = list(degree = 3)))
install.packages("e1071")
library(e1071)
result <- tune(svm, Species ~ ., data = iris, gamma = 2^(-1 : 1), cost = 2^(2 : 4))
attributes(result)
result$best.parameters


# 6. Class Imbalance ----
library(mlbench)
data("BreastCancer")
table(BreastCancer$Class) # benign - 양성, mlignant - 악성
# Up Sampling, Down Sampling ----
# caret::upSample(x, y), caret::downSample(x, y) ----
library(caret)
x <- upSample(subset(BreastCancer, select = -Class), BreastCancer$Class)
table(BreastCancer$Class)
table(x$Class)
NROW(x); NROW(unique(x))
library(party)
library(rpart)
data <- subset(BreastCancer, select = -Id)
parts <- createDataPartition(data$Class, p = .8) # 80% train data, 20% test data
data_train <- data[parts$Resample1, ]
data_test <- data[-parts$Resample1, ]
m <- rpart(Class ~ ., data = data_train)
confusionMatrix(data_test$Class, predict(m, newdata = data_test, type = "class"))

data_up_train <- upSample(subset(data_train, select = -Class), data_train$Class)
m <- rpart(Class ~ ., data = data_up_train)
confusionMatrix(data_test$Class, predict(m, newdata = data_test, type = "class"))

# SMOTE(Synthetic Minority Oversampling Technique) ----
# 1. 분류 개수가 적은 쪽의 데이터의 샘플을 취한 뒤 이 샘플의 k 최근접 이웃을 찾는다.
# 2. 현재 샘플과 이들 k개 이웃 간의 차를 구하고, 이 차이에 0 ~ 1 사이의 임의의 값을 곱하여 원래 샘프에 더한다.
# 3. 만든 새로운 샘플을 훈련 데이터에 추가한다.
# DMwR::SMOTE(form, data, perc.over = 200, k = 5, perc.under = 200) ----
## perc.over : 적은 쪽의 데이터를 얼마나 추가로 샘플링해야 하는지
## k : 고려할 최근접 이웃의 수
## perc.under : 적은 쪽의 데이터를 추가로 샘플링할 때 
##              각 샘플에 대응해서 많은 쪽의 데이터를 얼마나 샘플링할지 지정
library(DMwR)
data(iris)
data <- iris[, c(1, 2, 5)]
data$Species <- factor(ifelse(data$Species == "setosa", "rare", "common"))
table(data$Species)
newData <- SMOTE(Species ~ ., data, perc.over = 600, perc.under = 100)
table(newData$Species)


# 7. caret Packages ----
# caret::train(form, data, ..., weights) ----
# caret::train(x, y, method = "rf", preProcess = NULL, weights = NULL, metric, trControl = trainControl(), ...) ----
## method - rf : randomForest ( names(getModelInfo()) ) / 사용할 기계 학습 모델
## preProcess - center : 평균이 0이 되게 한다. / 수행할 데이터 전처리
##            - scale : 분산이 1이 되게 한다.
## metric : 분류 문제의 경우 정확도(accuracy), 회귀 문제의 경우 RMSE로 자동 지정
# caret::trainControl(method = "boot", number, repeats, p = 0.75) ----
# method : Data sampling technique - boot, boot632, cv(cross validation), repeatedcv,
#                                    LOOCV(Leave One Out Cross Validation)
# number : Cross validation을 몇 겹으로 할 것인지 or Bootstrapping을 몇 회 수행할 것인지 지정
# repeats : number of data sampling repetitions
## Bootstrapping ----
x <- rnorm(1000, mean = 30, sd = 3)
t.test(x)
library(foreach)
bootmean <- foreach(i = 1 : 10000, .combine = c) %do% {
  return(mean(x[sample(1 : NROW(x), replace = T)]))
}
bootmean <- sort(bootmean)
bootmean[c(10000 * 0.025, 10000 * 0.975)]
##
library(caret)
set.seed(137)
train_idx <- createDataPartition(iris$Species, p = 0.8)[[1]]
data_train <- iris[train_idx, ]
data_test <- iris[-train_idx, ]
(m <- train(Species ~ ., data = data_train, preProcess = c("pca"),
            method = "rf", ntree = 1000, trControl = trainControl(method = "cv",
                                                                  number = 10,
                                                                  repeats = 3)))
confusionMatrix(predict(m, newdata = data_test, type = "raw"), data_test$Species)
