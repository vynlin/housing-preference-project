# Run from the repo root, or adjust the path below
df <- read.csv("data/Responses.csv", stringsAsFactors = FALSE)

#rename variables columns for easier coding later
colnames(df) <- c(
  "timestamp",
  "dorm",              # "I am currently living in HKU dorm..."
  "student_status",    # "I am an"
  "year",              # "Year of study"
  "faculty",           # "My faculty is"
  "housing_fee",       # "I am currently paying (or willing to pay)..."
  "scholarship",       # "I am currently receiving"
  "price",             # "Price/Affordability"
  "commute",           # "Commute time to campus"
  "privacy",           # "Privacy"
  "study_env",         # "Study environment and quiet hours"
  "social",            # "Social life and community building"
  "facilities",        # "Facilities and amenities..."
  "rules",              # "Rules and flexibility..."
  "sustainability",   # NEW Q14: "Sustainability features (energy, recycling, water)" (1-5)
  "green_premium"    # NEW Q15: "Willingness to pay a premium for green housing" (1-5)
)
str(df)
#convert DV to binary format
df$dorm <- ifelse(df$dorm == "Yes", 1, 0)
table(df$dorm)  # Check

#CONVERT CATEGORICAL VARIABLES TO FACTORS
df$student_status <- factor(df$student_status,
                            levels = c("Local student", "Mainland student", "International student"),
                            labels = c("Local", "Mainland", "International"))
df$year <- factor(df$year)
df$faculty <- factor(df$faculty)
df$housing_fee <- factor(df$housing_fee,
                         levels = c("Less than HK$2,000",
                                    "HK$2,000 - HK$3,999",
                                    "HK$4,000 - HK$5,999",
                                    "HK$6,000 - HK$7,999",
                                    "HK$8,000 or more"),
                         ordered = TRUE)# (preserve order)

df$scholarship <- factor(df$scholarship)

#CHECK FOR MISSING VALUES
colSums(is.na(df))  # Check missing values per column
# Remove rows with any NA
df_clean <- na.omit(df)
# Verify
nrow(df)        # Original: should be 119
nrow(df_clean)  # After cleaning

#REMOVE TIMESTAMP & PREPARE FINAL DATASET
df_final <- df_clean[, -which(names(df_clean) == "timestamp")]

str(df_final )    # View structure
summary(df_final ) # Summary statistics

#SPLIT INTO TRAIN/TEST (75% / 25%)
library(caTools)
set.seed(502)  
spl <- sample.split(df_final$dorm, SplitRatio = 0.75)
table(spl)  # Should show count of TRUE (training) and FALSE (testing)
train_data <- subset(df_final, spl == TRUE)
test_data <- subset(df_final, spl == FALSE)
nrow(train_data)  # Should be ~75% of total (approximately 89-90 rows)
nrow(test_data) 

# BUILD LOGISTIC REGRESSION MODEL

logit1 <- glm(dorm~ student_status + year + faculty + housing_fee + 
                     scholarship + price + commute + privacy + study_env + 
                     social + facilities + rules,
                   data = train_data,
                   family = binomial(link = "logit"))# 1st model with all IVs

# Sustainable pref
logit1_sustain <- glm(dorm ~ student_status + year + faculty + housing_fee +
                      scholarship + price + commute + privacy + study_env +
                      social + facilities + rules + sustainability + green_premium,
                      data = train_data,
                      family = binomial(link = "logit"))
summary(logit1_sustain)

summary(logit1) #Interpret results: price (p = 0.0371*) → SIGNIFICANT! Price importance increases odds of living in dorm, year4 (p = 0.0754.) → Marginally significant (Year 4 students less likely to live in dorm) Everything else: Not significant (p > 0.1)

library(car)#check for MULTICOLLINEARITY problem
vif(logit1)  #All good (<10)

test_data$pred_prob <- predict(logit1, newdata = test_data, type = "response") #Perform on test data
test_data$pred_class <- ifelse(test_data$pred_prob > 0.5, 1, 0)
table(Predicted = test_data$pred_class, Actual = test_data$dorm)# Confusion matrix

accuracy <- sum(test_data$pred_class == test_data$dorm) / nrow(test_data)# out of sample Accuracy
print(paste("Accuracy:", round(accuracy, 3)))
 
#PROBLEM,HELP RESOLVE PLS:
library(pROC)#old (current method)
ROCplot <- roc(test_data$dorm, test_data$pred_prob) #ROC CURVE 
plot(ROCplot, main = "ROC Curve", col = "blue", lwd = 2)
auc(ROCplot)  # AUC Should be > 0.7 for good model, so logit1 AUC is not very good 
#Model with all variables is performing bad on test set (overall accuracy of only .414, so its pred ability is even worse than random guessing)

library(ROCR)#slide update 
ROCRpred <- prediction(test_data$pred_prob, test_data$dorm)
ROCCurve <- performance(ROCRpred, "tpr", "fpr")
plot(ROCCurve, colorize=TRUE, print.cutoffs.at=seq(0,1,0.1),
     text.adj=c(-0.2, 1)) # third argument defines thresholds shown,last argument defines the position of the threshold values
AUC_rocr <- as.numeric(performance(prediction(test_data$pred_prob, factor(test_data$dorm, levels = c(0, 1))), "auc")@y.values)
print(AUC_rocr)#slide: as.numeric(performance(ROCRpred, "auc")@y.values)= 1- AUC(pROC) 

if (AUC_rocr < 0.5) AUC_rocr <- 1 - AUC_rocr #flip manually if still <0.5 as it's maybe some bug in the packages? So why just DON'T use this in the first place. USE CURRENT pROC pls
print(AUC_rocr)

#Improve model progress

logit2 <- glm(dorm ~ student_status + year + housing_fee + 
                scholarship + price + commute + privacy + study_env + 
                social + facilities + rules,
              data = train_data, family = binomial) # 2nd model with all IVs excluding faculty (too many categories)
summary(logit2)
test_data$pred2 <- predict(logit2, newdata = test_data, type = "response")
roc2 <- roc(test_data$dorm, test_data$pred2)
auc(roc2)

# FUTURE MODEL (uncomment once sustainability + green_premium columns exist):
# logit2_sustain <- glm(dorm ~ student_status + year + housing_fee +
#                 scholarship + price + commute + privacy + study_env +
#                 social + facilities + rules + sustainability + green_premium,
#               data = train_data, family = binomial)
# summary(logit2_sustain)
# test_data$pred2_sustain <- predict(logit2_sustain, newdata = test_data, type = "response")
# roc2_sustain <- roc(test_data$dorm, test_data$pred2_sustain)
# auc(roc2_sustain)  # compare against auc(roc2) to see if the new questions add predictive value


logit_simple <- glm(dorm ~ price + year,
                    data = train_data,
                    family = binomial)# 3rd Model kept only significant IVs: price and year
summary(logit_simple)

pred_simple <- predict(logit_simple, newdata = test_data, type = "response")# Test performance on test set
roc_simple <- roc(test_data$dorm, pred_simple)
auc(roc_simple)
table(test_data$dorm, pred_simple > 0.5)#Confusion matrix
#overall Accuracy= (5+13) / 29 =62.1%(Better than random 50%); Sensitivity(TPR) = 13/16 =81.25% (Good at identifying dorm students); Specificity(Tnr) = 5/13 =38.46% (Poor at identifying off-campus)

# Tried adding an interaction term (like in lecture), but didn't get a better model
logit4 <- glm(dorm ~ student_status * scholarship + social + commute,
              data = train_data, family = binomial)
summary(logit4)
test_data$pred_prob4 <- predict(logit4, newdata = test_data, type = "response")
roc4 <- roc(test_data$dorm, test_data$pred_prob4)
auc(roc4)

# FINAL MODEL FOR REPORT
summary(logit_simple)  # Predict house preference by price and year of study as IVs

# Model interpretation:
# 1. Price importance (p=0.005**) - Significant
# 2. Year 4 (p=0.0956.) - Marginally significant
# 3. AIC = 119.08 - Best among all models tested

#ACKNOWLEDGE SHORTCOMINGS & LIMITATIONS
#1. Sample size: N=117 (small, limits generalizability)
#2. Model AUC: 0.47 (below 0.7 threshold for good prediction)
#3. Missing predictors: family income, distance from home, accommodation quality
#4. Class imbalance: Model biased toward majority class (dorm residents)
#5. Only 2/11 predictors significant: housing choice may be more complex
#6. Single train/test split: results may vary with different data splits

#RECOMMENDATIONS FOR FUTURE RESEARCH
#1. Collect larger sample (target 300+)
#2. Add socioeconomic factors (family income, financial independence)
#3. Include distance/location data (proximity to home)
#4. Use k-fold cross-validation for robust evaluation
#5. Consider non-linear models or machine learning approaches
