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

# with, within(data, expr, ...) : data - environment to make data ----
#                                 expr - expression to evaluate
#                                 ...  - function parameter
# within - can modify data
print(mean(iris$Sepal.Length))
print(mean(iris$Sepal.Width))
with(iris, {
  print(mean(Sepal.Length))
  print(mean(Sepal.Width))
})
x <- data.frame(val = c(1, 2, 3, 4, NA, 5, NA)); x
x <- within(x, {
  val <- ifelse(is.na(val), median(val, na.rm = T), val)
}); x
x$val[is.na(x$val)] <- median(x$val, na.rm = T); x
data(iris)
iris[1, 1] = NA; iris
median_per_species <- sapply(split(iris$Sepal.Length, iris$Species), 
                             median, na.rm = T)
iris <- within(iris, {
  Sepal.Length <- ifelse(is.na(Sepal.Length), 
                         median_per_species[Species],
                         Sepal.Length)
}); iris

# attach, detach, search ----
Sepal.Width
attach(iris); head(Sepal.Width)
detach(iris); Sepal.Width
search()
attach(iris); search()
detach(iris); search()
# warning : modify after attach, then not be applied on data after detach
head(iris); attach(iris)
Sepal.Width[1] = -1
Sepal.Width
detach(iris); head(iris)

# which(.max / .min) : find right index ----
subset(iris, Species == "setosa")
iris[iris$Species == "setosa", ]
which(iris$Species == "setosa")
which.min(iris$Sepal.Length)
which.max(iris$Sepal.Length)

# aggregate(X, by, FUN) : X - R Instance ----
#                         by - list that bind group
#                         FUN - function
# aggregate(formula, data, FUN) : formula - y ~ x shape
#                                           y -> value for calculating
#                                           x -> standard value to bind data
aggregate(Sepal.Width ~ Species, iris, mean)
tapply(iris$Sepal.Length, iris$Species, mean)

# stack / unstack ----
x <- data.frame(a = c(3, 2, 9),
                b = c(5, 3, 2),
                c = c(4, 5, 7))
x_stacked <- stack(x)
summaryBy(values ~ ind, x_stacked)
unstack(x_stacked, values ~ ind)

# Packages ----
# sqldf ----
install.packages("sqldf")
library(sqldf)
sqldf("select distinct Species from iris")
# warning : SQL not use '.' -> _
sqldf("select avg(Sepal_Length) from iris where Species = 'setosa'")
mean(subset(iris, Species == "setosa")$Sepal.Length)
sqldf("select species, avg(sepal_length) from iris group by species")
sapply(split(iris$Sepal.Length, iris$Species), mean)

# plyr ----
# a : array / d : data.frame / l : list
install.packages("plyr")
library(plyr)
# adply ----
# apply returns one data type
apply(iris[, 1 : 4], 1, function(row){ print(row) })
apply(iris, 1, function(row){ print(row) })
adply(iris, 1, function(row){ row$Sepal.Length >= 5.0 &
                              row$Species == "setosa"})
# ddply ----
ddply(iris, .(Species), function(sub){
  data.frame(sepal.width.mean = mean(sub$Sepal.Width))})
ddply(iris, .(Species, Sepal.Length > 5.0), function(sub){
  data.frame(sepal.width.mean = mean(sub$Sepal.Width))})
head(baseball)
help(baseball)
head(subset(baseball, id == "ansonca01"))
ddply(baseball, .(id), function(sub){ mean(sub$g) })
# mdplyr ----
x <- data.frame(mean = 1 : 5, sd = 1 : 5); x
mdply(x, rnorm, n = 2)

# base ----
# transform ----
head(ddply(baseball, .(id), transform, cyear = year - min(year) + 1))
# mutate : improve transform ----
head(ddply(baseball, .(id), mutate, 
           cyear = year - min(year) + 1, log_cyear = log(cyear)))
# summarise ----
head(ddply(baseball, .(id), summarise, minyear = min(year)))
head(ddply(baseball, .(id), summarise, minyear = min(year), maxyear = max(year)))
# subset ----
head(ddply(baseball, .(id), subset, g == max(g)))

# reshape2 ----
install.packages("reshape2")
library(reshape2)
# melt ----
head(french_fries)
m <- melt(french_fries, id.vars = 1 : 4)
head(m)
library(plyr)
ddply(m, .(variable), summarise, mean = mean(value, na.rm = T))
french_fries[!complete.cases(french_fries), ]
m <- melt(id = 1 : 4, french_fries, na.rm = T)
head(m)
# cast ----
# formula rule
# 1) id ~ variable
# 2) use '.' if don't specify all variables
# 3) use '...' to express the others
m <- melt(french_fries, id.vars = 1 : 4); m
r <- dcast(m, time + treatment + subject + rep ~ ...)
rownames(r) <- NULL
rownames(french_fries) <- NULL
identical(r, french_fries)
# summarise data
m <- melt(french_fries, id.vars = 1 : 4); head(m)
dcast(m, time ~ variable)
dcast(m, time ~ variable, mean, na.rm = T)
dcast(m, time ~ treatment + variable, mean, na.rm = T)
ddply(m, .(time, treatment, variable), function(rows){
  return(mean(rows$value, na.rm = T))
}); head(m)

# data.table ----
install.packages("data.table")
library(data.table)
iris_table <- as.data.table(iris)
x <- data.table(x = c(1, 2, 3), y = c("a", "b", "c"))
class(data.table())
tables()
DT <- as.data.table(iris)
DT[1, ]; DT[DT$Species == "setosa", ]
DT[1, Sepal.Length] # Not "Sepal.Length"
DT[1, list(Sepal.Length, Species)] # Not DT[1, c(Sepal.Length, Species)]
DT[, mean(Sepal.Length)]
DT[, mean(Sepal.Length - Sepal.Width)]
DT <- as.data.table(iris)
head(iris)
iris[1, 1]
DT[1, 1] # row 1 and one
DT[1, 1, with = FALSE] # 1 -> column number
iris[1, c("Sepal.Length")]
DT[1, c("Sepal.Length")]
DT[1, c("Sepal.Length"), with = F]
DT[, mean(Sepal.Length), by = "Species"]
DT <- data.table(x = c(1, 2, 3, 4, 5),
                 y = c("a", "a", "a", "b", "b"),
                 z = c("c", "c", "d", "d", "d")); DT
DT[, mean(x), by = "y,z"]
# setkey ----
DF <- data.frame(x = runif(26000), y = rep(LETTERS, each = 10000))
str(DF)
head(DF)
system.time(x <- DF[DF$y == "C", ])
DT <- as.data.table(DF)
setkey(DT, y)
system.time(x <- DT[J("C"), ])
DT[J("C"), mean(x)]
DT[J("C"), list(x_mean = mean(x), x_std = sd(x))]
DT1 <- data.table(x = runif(260000),
                  y = rep(LETTERS, each = 10000))
DT2 <- data.table(y = c("A", "B", "C"), z = c("a", "b", "c"))
setkey(DT1, y)
system.time(DT1[DT2, ])
DF1 <- as.data.frame(DT1)
DF2 <- as.data.frame(DT2)
system.time(merge(DF1, DF2))
# list -> data.frame ----
library(plyr)
system.time(x <- ldply(1 : 10000, function(x){
  data.frame(val = x,
             val2 = 2 * x,
             val3 = 2 / x,
             val4 = 4 * x,
             val5 = 4 / x)
}))
system.time(x <- llply(1 : 10000, function(x){
  data.frame(val = x,
             val2 = 2 * x,
             val3 = 2 / x,
             val4 = 4 * x,
             val5 = 4 / x)
}))
x <- lapply(1 : 10000, function(x){
  data.frame(val = x,
             val2 = 2 * x,
             val3 = 2 / x,
             val4 = 4 * x,
             val5 = 4 / x)
})
system.time(y <- do.call(rbind, x))
# rbindlist ----
system.time(x <- ldply(1 : 10000, function(x){
  data.frame(val = x,
             val2 = 2 * x,
             val3 = 2 / x,
             val4 = 4 * x,
             val5 = 4 / x)
}))
system.time(x <- llply(1 : 10000, function(x){
  data.frame(val = x,
             val2 = 2 * x,
             val3 = 2 / x,
             val4 = 4 * x,
             val5 = 4 / x)
}))
system.time(x <- rbindlist(x))
head(x)

# foreach ----
install.packages("foreach")
library(foreach)
foreach(i = 1 : 5) %do% {
  return(i)
}
foreach(i = 1 : 5, .combine = c) %do% {
  return(i)
}
foreach(i = 1 : 10, .combine = "+") %do% {
  return(i)
}

# Graph ----
methods("plot")
# plot(x, y) ----
install.packages("mlbench")
library(mlbench)
library(help = "mlbench")
data(Ozone)
plot(Ozone$V8, Ozone$V9)
# graph options ----
# 1. xlab, ylab (name of x, y axis)
# 2. main (name of graph)
# 3. pch (type of dot)
# 4. cex (size of dit)
# 5. col (color)
# 6. xlim, ylim (range of x, y axis)
# 7. type (type of graph[p : dot, l : line, b : line and dot, n : nothing])
plot(Ozone$V8, Ozone$V9, 
     xlab = "Sandburg Temperature", ylab = "El Monte Temperature")
plot(Ozone$V8, Ozone$V9, 
     xlab = "Sandburg Temperature", ylab = "El Monte Temperature",
     main = "Ozone")
plot(Ozone$V8, Ozone$V9, 
     xlab = "Sandburg Temperature", ylab = "El Monte Temperature",
     main = "Ozone", pch = 20)
plot(Ozone$V8, Ozone$V9, 
     xlab = "Sandburg Temperature", ylab = "El Monte Temperature",
     main = "Ozone", pch = "+")
example(points)
plot(Ozone$V8, Ozone$V9, 
     xlab = "Sandburg Temperature", ylab = "El Monte Temperature",
     main = "Ozone", cex = .1)
plot(Ozone$V8, Ozone$V9, 
     xlab = "Sandburg Temperature", ylab = "El Monte Temperature",
     main = "Ozone", col = "#FF0000")
min(Ozone$V8, na.rm = T)
min(Ozone$V9, na.rm = T)
max(Ozone$V8, na.rm = T)
max(Ozone$V9, na.rm = T)
plot(Ozone$v8, Ozone$v9, 
     xlab = "Sandburg Temperature", ylab = "El Monte Temperature",
     main = "Ozone",
     xlim = c(0, 100), ylim = c(0, 90))
data(cars)
str(cars)
head(cars)
plot(cars)
plot(cars, type = "l")
plot(cars, type = "b")
plot(cars, type = "o") # overlapped
tapply(cars$dist, cars$speed, mean)
plot(tapply(cars$dist, cars$speed, mean), type = "o",
     cex = 0.5, xlab = "speed", ylab = "dist")
# lty : 0 - blank, 1 - solid, 2 - dashed, 3 - dotted, 
#       4 - dotdash, 5 - longdash, 6 - twodash
plot(cars, type = "l", lty = "dashed")
# jitter ----
head(Ozone[, c("V6", "V7")])
plot(Ozone$V6, Ozone$V7, xlab = "Windspeed", ylab = "Humidity",
     main = "Ozone", pch = 20, cex = .5)
plot(jitter(Ozone$V6), jitter(Ozone$V7), xlab = "Windspeed", ylab = "Humidity",
     main = "Ozone", pch = 20, cex = .5)
# Type of Graph ----
# points ----
plot(iris$Sepal.Width, iris$Sepal.Length, cex = .5, pch = 20, xlab = "width",
     ylab = "length", main = "iris")
points(iris$Petal.Width, iris$Petal.Length, cex = .5, pch = "+", col = "#FF0000")
with(iris, {
  plot(NULL, xlim = c(0, 5), ylim = c(0, 10),
       xlab = "width", ylab = "length", main = "iris", type = "n")
  points(Sepal.Width, Sepal.Length, cex = .5, pch = 20)
  points(Petal.Width, Petal.Length, cex = .5, pch = 20, col = "#FF0000")
})
# lines ----
x <- seq(0, 2 * pi, 0.1)
y <- sin(x)
plot(x, y, cex = .5, col = "red")
lines(x, y)
data(cars)
head(cars)
plot(cars)
lines(lowess(cars)) # LOWESS find low degree polynomial like 'y=ax+b' or 'y=ax^2+bx+c'
                    # LOWESS is called 'Locally Weighted Polynomial Regression'
# abline ----
# draw y=ax+b or y=h or x=v straight line
plot(cars, xlim = c(0, 25))
abline(a = -5, b = 3.5, col = "red")
plot(cars, xlim = c(0, 25))
abline(a = -5, b = 3.5, col = "red")
abline(h = mean(cars$dist), lty = 2)
abline(v = mean(cars$speed), lty = 2)
# curve ----
curve(sin, 0, 2 * pi)
# polygon ----
m <- lm(dist ~ speed, data = cars) # dist = 3.932 * speed - 17.5791 + e
plot(cars)
abline(m)
p <- predict(m, interval = "confidence")
head(p)
m <- lm(dist ~ speed, data = cars)
p <- predict(m, interval = "confidence")
x <- c(cars$speed, tail(cars$speed, 1), rev(cars$speed), cars$speed[1])
y <- c(p[, "lwr"], tail(p[, "upr"], 1), rev(p[, "upr"]), p[, "lwr"][1])
plot(cars)
abline(m)
polygon(x, y, col = rgb(.7, .7, .7, .5))
# text(x, y, labels, adj, pos. ...) ----
# labels = seq_along(x) - 1, 2, 3 ... NROW(x)
# priority : 1) pos, 2) adj
# adj : (0, 0) - top right, (0, 1) - bottom right
#       (1, 0) - top left, (1, 1) - bottom left
# pos : 1 - bottom, 2 - left, 3 - top, 4 - right
plot(4 : 6, 4 : 6)
text(5, 5, "X")
text(5, 5, "00", adj = c(0, 0))
text(5, 5, "01", adj = c(0, 1))
text(5, 5, "10", adj = c(1, 0))
text(5, 5, "11", adj = c(1, 1))
plot(cars, cex = .5)
text(cars$speed, cars$dist, pos = 4)
# identify ----
plot(cars, cex = .5)
identify(cars$speed, cars$dist)
# legend ----
plot(iris$Sepal.Width, iris$Sepal.Length, pch = 20, xlab = "width", ylab = "length")
points(iris$Petal.Width, iris$Petal.Length, pch = "+", col = "#FF0000")
legend("topright", legend = c("Sepal", "Petal"), pch = c(20, 43),
       col = c("black", "red"), bg = "gray")
# matplot ----
x <- seq(-2 * pi, 2 * pi, 0.01); x
y <- matrix(c(cos(x), sin(x)), ncol = 2); y
matplot(x, y, lty = c("solid", "dashed"), cex = .2, type = "l")
abline(h = 0, v = 0)
# boxplot(formula, data, horizontal, notch) ----
boxplot(iris$Sepal.Width)
boxstats <- boxplot(iris$Sepal.Width); boxstats
boxstats <- boxplot(iris$Sepal.Width, horizontal = T);
text(boxstats$out, rep(1, NROW(boxstats$out)), labels = boxstats$out,
     pos = c(1, 1, 3, 1))
sv <- subset(iris, Species == "setosa" | Species == "versicolor")
sv$Species <- factor(sv$Species)
boxplot(Sepal.Width ~ Species, data = sv, notch = T)
