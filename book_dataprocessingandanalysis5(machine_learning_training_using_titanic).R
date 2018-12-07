# 1. Titanic Data form ----
## pclass : 1, 2, 3 등석 정보
## survived : 생존 여부(survived, dead)
## name : 이름
## sex : 성별(female, male)
## age : 나이
## sibsp : 함께 탑승한 형제 or 배우자의 수
## parch : 함께 탑승한 부모 or 자녀의 수
## ticket : 티켓 번호
## fare : 티켓 요금
## cabin : 선실 번호
## embarked : 탑승한 곳(Cherbourg, Queenstown, Southapmton)


# 2. Load data ----
titanic <- read.csv("titanic3.csv")
titanic <- titanic[, !names(titanic) %in% c("home.dest", "boat", "body")]
str(titanic)

# (1) Data type Specification
titanic$pclass <- as.factor(titanic$pclass)
titanic$name <- as.character(titanic$name)
titanic$ticket <- as.character(titanic$ticket)
titanic$cabin <- as.character(titanic$cabin)
titanic$survived <- factor(titanic$survived, levels = c(0, 1), labels = c("dead", "survived"))
str(titanic)
levels(titanic$embarked)
table(titanic$embarked)
levels(titanic$embarked)[1] <- NA
table(titanic$embarked, useNA = "always") # table()은 NA 값을 제외하기때문.
titanic$cabin <- ifelse(titanic$cabin == "", NA, titanic$cabin)
str(titanic)

# (2) Division of Test Data
library(caret)
set.seed(137)
test_idx <- createDataPartition(titanic$survived, p = 0.1)$Resample1
titanic_test <- titanic[test_idx, ]
titanic_train <- titanic[-test_idx, ]
NROW(titanic_test)
prop.table(table(titanic_test$survived))
NROW(titanic_train)
prop.table(table(titanic_train$survived))
save(titanic, titanic_test, titanic_train, file = "titanic.RData")

# (3) Ready for Cross Validation
createFolds(titanic_train$survived, k = 10)
create_ten_fold_cv <- function(){
  set.seed(137)
  lapply(createFolds(titanic_train$survived, k = 10), function(idx){
    return(list(train = titanic_train[-idx, ],
                validation = titanic_train[idx, ]))
  })
}
x <- create_ten_fold_cv()
str(x)
head(x$Fold01$train) # x[[1]]$train


# 3. Data Searching ----
library(Hmisc)
data <- create_ten_fold_cv()[[1]]$train
# reverse는 종속 변수(lhs)에 따라 독립 변수들을 분할하여 보여준다.
summary(survived ~ pclass + sex + age + sibsp + parch + fare + embarked, 
        data = data, method = "reverse")
data_complete <- data[complete.cases(data), ]
featurePlot(data_complete[, sapply(names(data_complete), function(n){
  is.numeric(data_complete[, n])
})], data_complete[, c("survived")], "ellipse")
mosaicplot(survived ~ pclass + sex, data = data, color = T, main = "pclass and sex")
xtabs(~ sex + pclass, data = data)
xtabs(survived == "survived" ~ sex + pclass, data = data)
xtabs(survived == "survived" ~ sex + pclass, data = data) / xtabs(~ sex + pclass, data = data)


# 4. Evaluation Metrics ----
## i will use 'Accuracy'
predicted <- c(1, 0, 0, 1, 1)
actual <- c(1, 0, 0, 0, 0)
sum(predicted == actual) / NROW(predicted)


# 5. Decision Tree Model ----
library(rpart)
(m <- rpart(survived ~ - name - ticket - cabin + ., data = titanic_train))
p <- predict(m, newdata = titanic_train, type = "class")
head(p)

# (1) Cross Validation of 'rpart'
library(rpart)
library(foreach)
folds <- create_ten_fold_cv()
rpart_result <- foreach(f=folds) %do% {
  model_rpart <- rpart(survived ~ pclass + sex + age + sibsp + parch + fare + embarked
                       , data = f$train)
  predicted <- predict(model_rpart, newdata = f$validation, type = "class")
  
  return(list(actual = f$validation$survived, predicted = predicted))
}
head(rpart_result)

# (2) Accuracy Evaluation
evaluation <- function(lst){
  accuracy <- sapply(lst, function(one_result){
    return(sum(one_result$predicted == one_result$actual) / NROW(one_result$actual))
  })
  print(sprintf("MEAN +/- SD: %.3f +/- %.3f",
                mean(accuracy), sd(accuracy)))
  return(accuracy)
}
(rpart_accuracy <- evaluation(rpart_result))

# (3) Conditional Inference Tree
# 1) 다른 모델링 기법을 적용
# 2) 데이터 내에 숨겨진 쓸 만한 다른 특징 값을 찾기
library(party)
ctree_result <- foreach(f = folds) %do% {
  model_ctree <- ctree(survived ~ pclass + sex + age + sibsp + parch + fare + embarked,
                       data = f$train)
  predicted <- predict(model_ctree, newdata = f$validation, type = "response")
  
  return(list(actual = f$validation$survived, predicted = predicted))
}
(ctree_accuracy <- evaluation(ctree_result))
plot(density(rpart_accuracy), main = "rpart VS ctree")
lines(density(ctree_accuracy), col = "red", lty = "dashed")


# 6. Discovery of Another Feature
# (1) Family Identification using ticket
View(titanic_train[order(titanic_train$ticket), 
                   c("ticket", "parch", "name", "cabin", "embarked")])
sum(is.na(titanic_train$ticket))
sum(is.na(titanic_train$embarked))
sum(is.na(titanic_train$cabin))

# (2) Survival Probability Prediction
family_result <- foreach(f = folds) %do% {
  f$train$type <- "T"
  f$validation$type <- "V"
  all <- rbind(f$train, f$validation)
  ctree_model <- ctree(survived ~ pclass + sex + age + sibsp + parch + fare + embarked,
                       data = f$train)
  all$prob <- sapply(predict(ctree_model, type = "prob", newdata = all),
                     function(result){ result[1] })
}

# (3) Grant Family ID
library(plyr)
family_idx <- 0
ticket_based_family_id <- ddply(all, .(ticket), function(rows){
  family_idx <<- family_idx + 1
  return(data.frame(family_id = paste0("TICKET_", family_idx)))
})
str(ticket_based_family_id)
head(ticket_based_family_id)
all <- adply(all, 1, function(row){
  family_id <- NA
  if(!is.na(row$ticket)){
    family_id <- subset(ticket_based_family_id, ticket == row$ticket)$family_id
  }
  return(data.frame(family_id = family_id))
})
str(all)

# (4) Merger of Family Member Survival Probability
all <- ddply(all, .(family_id), function(rows){
  rows$avg_prob <- mean(rows$prob)
  return(rows)
})
all <- ddply(all, .(family_id), function(rows){
  rows$maybe_parent <- FALSE
  rows$maybe_child <- FALSE
  if(NROW(rows) == 1 || sum(rows$parch) == 0 || NROW(rows) == sum(is.na(rows$age))){
    return(rows)
  }
  
  max_age <- max(rows$age, na.rm = T)
  min_age <- min(rows$age, na.rm = T)
  return(adply(rows, 1, function(row){
    if(!is.na(row$age) && !is.na(row$sex)){
      row$maybe_parent <- (max_age - row$age) < 10
      row$maybe_child <- (row$age - min_age) < 10
    }
    return(row)
  }))
})
all <- ddply(all, .(family_id), function(rows){
  rows$avg_parent_prob <- rows$avg_prob
  rows$avg_child_prob <- rows$avg_prob
  if(NROW(rows) == 1 || sum(rows$parch) == 0){
    return (rows)
  }
  parent_prob <- subset(rows, maybe_parent == TRUE)[, "prob"]
  if(NROW(parent_prob) > 0){
    rows$avg_parent_prob <- mean(parent_prob)
  }
  child_prob <- c(subset(rows, maybe_child == T)[, "prob"])
  if(NROW(child_prob) > 0){
    rows$avg_child_prob <- mean(child_prob)
  }
  return(rows)
})

# (5) ctree() Modeling using Family Information
str(all)
f$train <- subset(all, type == "T")
f$validation <- subset(all, type == "V")

(m <- ctree(survived ~ pclass + sex + age + sibsp + parch + fare + embarked + maybe_parent +
              maybe_child + avg_prob + avg_parent_prob + avg_child_prob, data = f$train))
predicted <- predict(m, newdata = f$validation)

# (6) Performance Evaluation
family_result <- foreach(f = folds) %do% {
  f$train$type <- "T"
  f$validation$type <- "V"
  all <- rbind(f$train, f$validation)
  ctree_model <- ctree(survived ~ pclass + sex + age + sibsp + parch + fare + embarked,
                       data = f$train)
  all$prob <- sapply(predict(ctree_model, type = "prob", newdata = all),
                     function(result){ result[1] })
  
  # 티켓 번호를 사용한 family_id
  family_idx <- 0
  ticket_based_family_id <- ddply(all, .(ticket), function(rows){
    family_idx <<- family_idx + 1
    return(data.frame(family_id = paste0("TICKET_", family_idx)))
  })
  all <- adply(all, 1, function(row){
    family_id <- NA
    if(!is.na(row$ticket)){
      family_id <- subset(ticket_based_family_id, ticket == row$ticket)$family_id
    }
    return(data.frame(family_id = family_id))
  })
  
  # avg_prob
  all <- ddply(all, .(family_id), function(rows){
    rows$avg_prob <- mean(rows$prob)
    return(rows)
  })
  
  # maybe_parent, maybe_child
  all <- ddply(all, .(family_id), function(rows){
    rows$maybe_parent <- FALSE
    rows$maybe_child <- FALSE
    if(NROW(rows) == 1 || sum(rows$parch) == 0 || NROW(rows) == sum(is.na(rows$age))){
      return(rows)
    }
    
    max_age <- max(rows$age, na.rm = T)
    min_age <- min(rows$age, na.rm = T)
    return(adply(rows, 1, function(row){
      if(!is.na(row$age) && !is.na(row$sex)){
        row$maybe_parent <- (max_age - row$age) < 10
        row$maybe_child <- (row$age - min_age) < 10
      }
      return(row)
    }))
  })
  
  # avg_parent_prob, avg_child_prob
  all <- ddply(all, .(family_id), function(rows){
    rows$avg_parent_prob <- rows$avg_prob
    rows$avg_child_prob <- rows$avg_prob
    if(NROW(rows) == 1 || sum(rows$parch) == 0){
      return (rows)
    }
    parent_prob <- subset(rows, maybe_parent == TRUE)[, "prob"]
    if(NROW(parent_prob) > 0){
      rows$avg_parent_prob <- mean(parent_prob)
    }
    child_prob <- c(subset(rows, maybe_child == T)[, "prob"])
    if(NROW(child_prob) > 0){
      rows$avg_child_prob <- mean(child_prob)
    }
    return(rows)
  })
  
  # ctree 모델
  f$train <- subset(all, type == "T")
  f$validation <- subset(all, type == "V")
  
  (m <- ctree(survived ~ pclass + sex + age + sibsp + parch + fare + embarked + maybe_parent +
                maybe_child + avg_prob + avg_parent_prob + avg_child_prob, data = f$train))
  predicted <- predict(m, newdata = f$validation)
  return(list(actual = f$validation$survived, predicted = predicted))
}
family_accuracy <- evaluation(family_result)


# 7. Parallelization of Cross Validation
# (1) Perform 3 repetition of 10-folds Cross Validation
createMultiFolds(titanic_train$survived, k = 10, times = 3)
create_three_ten_fold_cv <- function(){
  set.seed(137)
  lapply(createMultiFolds(titanic_train$survived, k = 10, times = 3), function(idx){
    return(list(train = titanic_train[idx, ], validation = titanic_train[-idx, ]))
  })
}
folds <- create_three_ten_fold_cv()
ctree_result <- foreach(f = folds) %do% {
  model_ctree <- ctree(survived ~ pclass + sex + age + sibsp + parch + fare + embarked,
                       data = f$train)
  predicted <- predict(model_ctree, newdata = f$validation, type = "response")
  return(list(actual = f$validation$survived, predicted = predicted))
}
(ctree_accuracy <- evaluation(ctree_result))

# (2) Paralleization using foreach() and %dopar%
system.time(ctree_result <- foreach(f = folds) %do% {
  model_ctree <- ctree(survived ~ pclass + sex + age + sibsp + parch + fare + embarked,
                       data = f$train)
  predicted <- predict(model_ctree, newdata = f$validation, type = "response")
  return(list(actual = f$validation$survived, predicted = predicted))
})
install.packages("doParallel")
library(doParallel)
registerDoParallel(cores = 4)
system.time(ctree_result <- foreach(f = folds) %dopar% {
  model_ctree <- ctree(survived ~ pclass + sex + age + sibsp + parch + fare + embarked,
                       data = f$train)
  predicted <- predict(model_ctree, newdata = f$validation, type = "response")
  return(list(actual = f$validation$survived, predicted = predicted))
})


# 8. Better Algorithm Development
# 1) 리서치 : 같은 문제를 여러 논문에서 연구된 경우
# 2) 더 나은 피처의 개발 : 타이타닉의 경우 '가족'의 개념을 꺼내 사용하니 모델 성능 항샹.
# 3) 모델 선택
# 4) 문서화