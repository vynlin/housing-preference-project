# Run from the repo root, or adjust the path below
df <- read.csv("data/Responses.csv", stringsAsFactors = FALSE)


colnames(df) <- c("timestamp", "dorm", "student_status", "year", "faculty", "housing_fee", 
                  "scholarship", "price", "commute", "privacy", "study_env", "social", 
                  "facilities", "rules"
                  ,"sustainability"  
                  ,"green_premium"   
                  )

# Outcome variable to factor (binary)
df$dorm <- as.factor(df$dorm)
# Categorical variables to factors (leave numeric predictors as numeric)
df$student_status <- factor(df$student_status, levels = c("Local student", "Mainland student", "International student"),
                            labels = c("Local", "Mainland", "International"))
df$year           <- as.factor(df$year)
df$faculty        <- as.factor(df$faculty)
df$housing_fee    <- factor(df$housing_fee,
                            levels = c("Less than HK$2,000", "HK$2,000 - HK$3,999", 
                                       "HK$4,000 - HK$5,999", "HK$6,000 - HK$7,999", 
                                       "HK$8,000 or more"),
                            ordered = TRUE)
df$scholarship    <- as.factor(df$scholarship)

# Remove missing values and timestamp column
df_clean <- na.omit(df)
df_final <- df_clean[, -which(names(df_clean) == "timestamp")]

# Split train/test sets
library(caTools)

set.seed(502)  
spl <- sample.split(df_final$dorm, SplitRatio = 0.75)
train_data <- subset(df_final, spl == TRUE)
test_data  <- subset(df_final, spl == FALSE)

# ---- CART ----
library(caret)
library(rpart)
library(rpart.plot)

nFolds  <- trainControl(method = "cv", number = 5)
cpRange <- expand.grid(.cp = seq(0.01, 0.3, 0.01))
set.seed(520)
train(dorm ~ student_status + year + faculty + housing_fee + 
                   scholarship + price + commute + privacy + study_env + 
                   social + facilities + rules, 
                 data = train_data, method = "rpart", 
                 trControl = nFolds, tuneGrid = cpRange)

cart_final <- rpart(dorm ~ student_status + year + faculty + housing_fee + 
                      scholarship + price + commute + privacy + study_env + 
                      social + facilities + rules, 
                    method = "class", cp = 0.02, data = train_data)
prp(cart_final)

cart_final_sustain <- rpart(dorm ~ student_status + year + faculty + housing_fee +
                      scholarship + price + commute + privacy + study_env +
                      social + facilities + rules + sustainability + green_premium,
                    method = "class", cp = 0.02, data = train_data)
prp(cart_final_sustain)

# CART predictions & performance
cart_pred  <- predict(cart_final, newdata = test_data, type = "class")
library(caret)
confusionMatrix(cart_pred, test_data$dorm)

# CART ROC & AUC
library(ROCR)
cart_probs <- predict(cart_final, newdata = test_data, type = "prob")[,2]
cart_pred_roc <- prediction(cart_probs, test_data$dorm)
cart_perf_roc <- performance(cart_pred_roc, "tpr", "fpr")
plot(cart_perf_roc)
cart_auc <- as.numeric(performance(cart_pred_roc, "auc")@y.values)
cart_auc

# CART-sustain predictions & performance (with sustainability + green_premium)
cart_pred_sustain <- predict(cart_final_sustain, newdata = test_data, type = "class")
confusionMatrix(cart_pred_sustain, test_data$dorm)
cart_probs_sustain <- predict(cart_final_sustain, newdata = test_data, type = "prob")[,2]
cart_pred_roc_sustain <- prediction(cart_probs_sustain, test_data$dorm)
cart_auc_sustain <- as.numeric(performance(cart_pred_roc_sustain, "auc")@y.values)
cart_auc_sustain

# ---- RANDOM FOREST ----
library(randomForest)
set.seed(520)
rf_model <- randomForest(dorm ~ student_status + year + faculty + housing_fee + 
                           scholarship + price + commute + privacy + study_env + 
                           social + facilities + rules, 
                         data = train_data, ntree = 500, nodesize = 25)
rf_pred <- predict(rf_model, newdata = test_data)
confusionMatrix(rf_pred, test_data$dorm)

# SUSTAINABILITY PREF MODEL 
 rf_model_sustain <- randomForest(dorm ~ student_status + year + faculty + housing_fee +
                            scholarship + price + commute + privacy + study_env +
                           social + facilities + rules + sustainability + green_premium,
                          data = train_data, ntree = 500, nodesize = 25)
 rf_pred_sustain <- predict(rf_model_sustain, newdata = test_data)
 confusionMatrix(rf_pred_sustain, test_data$dorm)
 importance(rf_model_sustain)  # check where sustainability/green_premium rank vs. price, commute, etc.

# Random Forest ROC & AUC
rf_probs <- predict(rf_model, newdata = test_data, type = "prob")[,2]
rf_pred_roc <- prediction(rf_probs, test_data$dorm)
rf_perf_roc <- performance(rf_pred_roc, "tpr", "fpr")
plot(rf_perf_roc)
rf_auc <- as.numeric(performance(rf_pred_roc, "auc")@y.values)
rf_auc


#RESULTS SUMMARY (re-run with sustainability + green_premium included, n=160):
#CART is completely unchanged by adding sustainability variables - identical tree, identical confusion
#matrix (14/4/6/16), identical accuracy (75%), sensitivity (0.70), specificity (0.80), and AUC (0.70).
#The tree never selects sustainability or green_premium at any split.
#Random Forest (baseline, no sustainability): accuracy 72.5%, sensitivity 0.80, specificity 0.65, AUC 0.75.
#Random Forest (with sustainability): accuracy also 72.5% (unchanged overall), but sensitivity/specificity
#shift to 0.75/0.70 - it trades off which group it identifies well rather than actually improving.
#Variable importance (Random Forest with sustainability, Mean Decrease Gini): faculty (2.86) and year (2.80)
#are the top two predictors, followed by housing_fee (2.51) and privacy (2.39). sustainability (0.33) and
#green_premium (0.58) both rank in the bottom half - green_premium above scholarship and rules, but
#sustainability itself is the second-lowest of all 13 predictors, ahead of only student_status.
#Bottom line: adding the two new sustainability questions changes nothing for CART and leaves Random
#Forest's accuracy flat while only reshuffling its errors. Both models, and the logistic regressions in
#01_logistic_regression.R, agree: self-reported sustainability preferences are not a meaningful driver of
#dorm-vs-off-campus housing choice in this sample. Housing budget, year of study, faculty, privacy, and
#commute remain the consistent, real drivers across every model tried.

