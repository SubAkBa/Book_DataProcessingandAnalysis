# add column and row
rbind(c(1, 2, 3), c(4, 5, 6))
x <- data.frame(id = c(1, 2), name = c("a", "b"), stringsAsFactors = F)
str(x)
y <- rbind(x, c(3, "c"))
y
cbind(c(1, 2, 3), c(4, 5, 6))
y <- cbind(x, greek = c("alpha", "beta"))
str(y)
y <- cbind(x, greek = c("alpha", "beta"), stringsAsFactors = F)
str(y)
# Formula ----
# X1 * X2 -> X1 + X2 + X1 : X2
# X1 : X2 -> X1 and X2 Interaction
# X1 | X2 -> group by X2 and apply Y ~ X1 each group

# apply function ----
# apply(X, MARGIN, FUN) : X - Array or Matrix ----
#                         MARGIN - 1 : Row Direction, 2 : Column Direction
#                         FUN - function
#                         return - Vector, Array, List
sum(1 : 55)
d <- matrix(1 : 9, ncol = 3)
d
apply(d, 1, sum)
apply(d, 2, sum)
head(iris)
apply(iris[, 1 : 4], 2, sum)
# lapply(X, FUN, ...) : X - Vector, List, Data.frame ----
#                       FUN - function
#                       ... - function parameter
#                       return - List
result <- lapply(1 : 3, function(x) {x * 2})
result
result[[1]]
unlist(result)
x <- list(a = 1 : 3, b = 4 : 6)
lapply(x, mean)
lapply(iris[, 1 : 4], mean)
d <- as.data.frame(matrix(unlist(lapply(iris[, 1 : 4], mean)),
                          ncol = 4, byrow = T))
names(d) <- names(iris[, 1 : 4])
d
data.frame(do.call(cbind, lapply(iris[, 1 : 4], mean)))
# sapply(X, FUN, ...) : Equal lapply ----
#                       return - Vector, Matrix (one data type)
lapply(iris[, 1 : 4], mean)
sapply(iris[, 1 : 4], mean)
class(sapply(iris[, 1 : 4], mean)) # "numeric" -> have numeric vector
x <- sapply(iris[, 1 : 4], mean)
x; as.data.frame(x)
sapply(iris, class)
y <- sapply(iris[, 1 : 4], function(x) { x > 3 })
class(y); head(y)
# tapply(X, INDEX, FUN, ...) : X - Vector ----
#                              INDEX - bind data index
#                              FUN - function
#                              ... - function parameter
#                              return - Array
tapply(1 : 10, rep(1, 10), sum)
tapply(1 : 10, 1 : 10 %% 2 == 1, sum)
tapply(iris$Sepal.Length, iris$Species, mean)
m <- matrix(1 : 8, ncol = 2, dimnames = list(c("spring", "summer", 
                                               "fall", "winter"),
                                             c("male", "female"))); m
# INDEX - (n, m) list n first and list m
tapply(m, list(c(1, 1, 2, 2, 1, 1, 2, 2),
               c(1, 1, 1, 1, 2, 2, 2, 2)), sum)
# mapply(FUN, ...) : FUN - function ----
#                    ... - function parameter
# Make random number function
# rnorm(n, mean = 0, sd = 1) : mean = mean, sd = sd, count = n, Normal distribution
# runinf(n, min = 0, max = 1) : min = min, max = max, count = n, Uniform distribution
# rpois(n, lambda) : lambda = lambda, count = n, Poisson distribution
# rexp(n, rate = 1) : lambda = rate, count = n, Exponential distribution
rnorm(10, 0, 1) # mean = 0, sd = 1, count = 10
mapply(rnorm, c(1, 2, 3), c(0, 10, 100), c(1, 1, 1))
mapply(mean, iris[, 1 : 4])

# library(doBy) ----
install.packages("doBy")
library(doBy)
# summaryBy(formula, data = parent.frame()) ----
summary(iris)
quantile(iris$Sepal.Length)
quantile(iris$Sepal.Length, seq(0, 1, by = 0.1))
summaryBy(Sepal.Width + Sepal.Length ~ Species, iris)
# orderBy(formula, data) ----
order(iris$Sepal.Width)
iris[order(iris$Sepal.Width), ]
iris[order(iris$Sepal.Length, iris$Sepal.Width), ]
orderBy(~ Sepal.Width, iris)
orderBy(~ Species + Sepal.Width, iris)
# sampleBy(formula, frac = 0.1, replace = F, data = parent.frame(), systematic = F) ----
# extract balanced data
sample(1 : 10, 5)
sample(1 : 10, 5, replace = T)
sample(1 : 10, 10)
iris[sample(NROW(iris), NROW(iris)), ]
sampleBy(~ Species, 0.1, data = iris)

# split(x, f) : x - Vector, data.frame ----
#               f - standard factor for split
#               return - list
split(iris, iris$Species)
lapply(split(iris$Sepal.Length, iris$Species), mean)
# subset(x, subset, select) ----
subset(iris, Species == "setosa")
subset(iris, Species == "setosa" & Sepal.Length > 5.0)
subset(iris, select = c(Sepal.Length, Species))
subset(iris, select = -c(Sepal.Length, Species))
iris[, !names(iris) %in% c("Sepal.Length", "Species")]
# merge(x, y, by, all) ----
x <- data.frame(name = c("a", "b", "c"), math = c(1, 2, 3))
y <- data.frame(name = c("c", "b", "a"), english = c(4, 5, 6))
merge(x, y)
cbind(x, y)
x <- data.frame(name = c("a", "b", "c"),
                math = c(1, 2, 3))
y <- data.frame(name = c("a", "b", "d"),
                english = c(4, 5, 6))
merge(x, y)
merge(x, y, all = T)
# sort(x, decreasing = F, na.last = NA) : na.last - T(na to last) ----
#                                                   F(na to first)
#                                                   NA(exclude NA)
x <- c(20, 11, 33, 50, 47)
sort(x)
sort(x, decreasing = T); x
# order : return - row index ----
x <- c(20, 11, 33, 50, 47)
order(x)
iris[order(iris$Sepal.Length), ]
iris[order(iris$Sepal.Length, iris$Petal.Length), ]
