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
