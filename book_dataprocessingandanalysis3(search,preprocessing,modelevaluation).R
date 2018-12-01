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
