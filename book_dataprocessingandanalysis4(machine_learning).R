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
