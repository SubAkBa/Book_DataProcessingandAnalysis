# Linear Regression ----
# (1) Simple Linear Regression - Y = B0 + B1 * X + E
# B0 : Intercept, B1 : X's coefficient, E : error
# lm(formula, data) ----
data(cars)
head(cars)
m <- lm(dist ~ speed, cars); m # dist = -17.759 + 3.932 * speed + E
# coef(model), fitted(model), residuals(model), confint(model), deviance(model) ----
coef(m)
fitted(m)[1 : 4] # predicted dist value
residuals(m)[1 : 4] # E = Y - ^Y(predicted value)
fitted(m)[1 : 4] + residuals(m)[1 : 4]
cars$dist[1 : 4]
confint(m) # confidence interval
deviance(m) # sigma(all residuals)^2

# predict(object, newdata, interval = c(default = "none", "confidence", "prediction")) ----
m <- lm(dist ~ speed, data = cars)
predict(m, newdata = data.frame(speed = 3))
coef(m)
-17.579095 + 3.932409 * 3
predict(m, newdata = data.frame(speed = 3), interval = "confidence")
predict(m, newdata = data.frame(speed = 3), interval = "prediction")

# model evaluation : summary() ----
summary(m) # R-squared : How much the model explains the variance of the data
# F-statistic : Statistically, how meaningful is the model.
#               How much the difference 
#               in the sum of squared differences is significant
#               dist = B0 + E(Reduced Model) and 
#               dist = B0 + B1 * speed + E(Full Model)
#               Finally, F-sta~ is Hypothesis test result about
#               H0 : B1 = 0, H1 : B1 is not 0.
## R-squared, Adjusted R-squared ----
## SST(Total Sum of Squares) - sigma(Yi - Ybar)^2
## SSR(Sum of Squares due to Regression) - sigma(^Yi - Ybar)^2
## Yi - value of i th dependent variable
## Ybar - mean of Yi
## ^Yi - predicted value of i th dependent variable by Linear Regression model
## if number of independent variable increases, R-squared becomes large.
## Therefore, Use Adjusted R-squared.
## SSE(Sum of Squares due to residual errors) = SST - SSR = sigma(Yi - ^Yi)^2
## R^adj = 1 - (SSE / (n - k - 1)) * (SST / (n - 1))
## n - number of data, k - number of independent variable

# Analysis of Variance : evaluating model or comparing with between some of models ----
# anova(object, ..) ----
anova(m)
full <- lm(dist ~ speed, data = cars); full
reduced <- lm(dist ~ 1, data = cars); reduced
anova(reduced, full)
# Diagnosing model graph ----
# plot.lm(x, which = c(1 : 3, 5)) ----
par(mfrow = c(2, 2))
plot(m)
plot(m, which = c(4, 6))
par(mfrow = c(1, 1))

# Visualizing Regression Line
plot(cars$speed, cars$dist)
abline(coef(m))
summary(cars$speed)
predict(m, newdata = data.frame(speed = seq(4.0, 25.0, .2)),
        interval = "confidence")
speed <- seq(min(cars$speed), max(cars$speed), .1)
ys <- predict(m, newdata = data.frame(speed = speed), interval = "confidence")
matplot(speed, ys, type = 'n')
matlines(speed, ys, lty = c(1, 2, 2), col = 1)

# Multiple Linear Regression ----
m <- lm(Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width, data = iris); m
summary(m)
m <- lm(Sepal.Length ~ ., data = iris); m
summary(m)

# Make design matrix
# model.matrix(object, data = environment(object)) ----
model.matrix(m)[c(1, 51, 101), ]
anova(m)

# Visualizing Multiple Linear Regression model
with(iris, plot(Sepal.Width, Sepal.Length,
                cex = .7,
                pch = as.numeric(Species)))
as.numeric(iris$Species)
m <- lm(Sepal.Length ~ Sepal.Width + Species, data = iris)
coef(m)
abline(2.25, 0.8, lty = 1)
abline(2.25 + 1.45, 0.8, lty = 2)
abline(2.25 + 1.94, 0.8, lty = 3)
legend("topright", levels(iris$Species), pch = 1 : 3, bg = "white")
levels(iris$Species)

# Expression of Quadratic Equation ----
# I(x) ----
x <- 1 : 1000
y <- x^2 + 3 * x + 5 + rnorm(1000)
lm(y ~ I(x^2) + x)

x1 <- 1 : 1000
x2 <- 3 * x1
y <- 3 * (x1 + x2) + rnorm(1000)
lm(y ~ I(x1 + x2))
lm(y ~ x1 + x2)

x <- 101 : 200
y <- exp(3 * x + rnorm(100))
lm(log(y) ~ x)
y <- log(x) + rnorm(100)
lm(y ~ log(x))

# Interaction ----
# ex) dist ~ speed + size + speed : size or dist ~ speed * size ----
data(Orange); Orange
with(Orange, plot(Tree, circumference, xlab = "tree", ylab = "circumference"))
# interaction.plot(x.factor, trace.factor, response) ----
with(Orange, interaction.plot(age, Tree, circumference))
Orange[, "fTree"] <- factor(Orange[, "Tree"], ordered = F)
m <- lm(circumference ~ fTree * age, data = Orange)
anova(m)
head(model.matrix(m))
mm <- model.matrix(m)
mm[, grep("age", colnames(mm))]

# Outlier : Externally Studentized Residual(ESR) ----
# ESR - divide Residuals into Standard Deviation of Residuals
# rstudent(model), car::outlierTest(model, ...) ----
install.packages("car")
library(car)
data(Orange)
m <- lm(circumference ~ age + I(age^2), data = Orange)
rstudent(m)
Orange <- rbind(Orange, data.frame(Tree = as.factor(c(6, 6, 6)),
                                   age = c(118, 484, 664),
                                   circumference = c(177, 50, 30)))
tail(Orange)
m <- lm(circumference ~ age + I(age^2), data = Orange)
outlierTest(m)

# Variables Selection ----
# step(object, scope, direction = c("both", "forward", "backward")) ----
library(mlbench)
data(BostonHousing)
m <- lm(medv ~ ., data = BostonHousing)
m2 <- step(m, direction = "both")
formula(m2)
# AIC(Akaikie Information Criteria) - The lower, The better

# leaps::regsubsets(x, data, 
#                   method = c("exhaustive", "forward", "backward", "seqrep"),
#                   nbest = 1)
#                   exhaustive - Search all models
#                   forward - add variable
#                   backward - delete variable
#                   seqrep(sequential replacment) - repeat adding and deleting variable
#                   nbest - get n best model
install.packages("leaps")
library(leaps)
library(mlbench)
data(BostonHousing)
m <- regsubsets(medv ~ ., data = BostonHousing)
summary(m)
plot(m, scale = "adjr2")
plot(m, scale = "bic")