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
# sustainability (p=0.955) and green_premium (p=0.966) are both far from significant, with
# near-zero coefficients (-0.012 and -0.008). AIC=177.5, which is WORSE than logit1's 173.51
# (lower AIC = better) - adding these two questions doesn't just fail to help, it actively
# makes the model fit worse by spending 2 degrees of freedom on noise. This matches the
# CART/Random Forest finding: sustainability preferences are not a meaningful predictor here.

summary(logit1) #Interpret results (n=160, up from 117 previously): year4 (p=0.039*) → SIGNIFICANT, Year 4 students far less likely to live in dorm; facultyArts (p=0.006**) and housing_fee^4 (p=0.009**) → also SIGNIFICANT; price is now only marginal (p=0.067, previously significant at p=0.037 on the smaller sample); several other faculty levels marginal (p~0.07-0.09). AIC=173.51.

library(car)#check for MULTICOLLINEARITY problem
vif(logit1)  #All GVIF^(1/(2*Df)) well under 2 (faculty highest at ~1.16 despite 9 categories) - no serious multicollinearity concern

test_data$pred_prob <- predict(logit1, newdata = test_data, type = "response") #Perform on test data
test_data$pred_class <- ifelse(test_data$pred_prob > 0.5, 1, 0)
table(Predicted = test_data$pred_class, Actual = test_data$dorm)# Confusion matrix

accuracy <- sum(test_data$pred_class == test_data$dorm) / nrow(test_data)# out of sample Accuracy
print(paste("Accuracy:", round(accuracy, 3)))
# Accuracy = 0.65 (26/40 correct) - a real improvement over the previous smaller-sample run

library(pROC)
ROCplot <- roc(test_data$dorm, test_data$pred_prob) #ROC CURVE 
plot(ROCplot, main = "ROC Curve", col = "blue", lwd = 2)
auc(ROCplot)  # AUC = 0.6875 - close to the 0.7 "good model" threshold, and a large jump up from
# the near-chance AUC (0.47) we saw on the smaller (n=117), imbalanced dataset. The larger,
# now-balanced sample (81 dorm / 81 non-dorm) appears to be the main driver of the improvement.

library(ROCR)#slide update 
ROCRpred <- prediction(test_data$pred_prob, test_data$dorm)
ROCCurve <- performance(ROCRpred, "tpr", "fpr")
plot(ROCCurve, colorize=TRUE, print.cutoffs.at=seq(0,1,0.1),
     text.adj=c(-0.2, 1)) # third argument defines thresholds shown,last argument defines the position of the threshold values
AUC_rocr <- as.numeric(performance(prediction(test_data$pred_prob, factor(test_data$dorm, levels = c(0, 1))), "auc")@y.values)
print(AUC_rocr)# = 0.6875, matching pROC exactly - no flip needed this time.

if (AUC_rocr < 0.5) AUC_rocr <- 1 - AUC_rocr #kept as a safety net, but not triggered on this run
print(AUC_rocr)
# NOTE ON THE OLD "PROBLEM": the flip was needed before because that run's model was
# genuinely performing below chance (AUC 0.47) on a small, imbalanced sample - not a package
# bug. With more data and a balanced outcome, pROC and ROCR now agree without any flip.

#Improve model progress

logit2 <- glm(dorm ~ student_status + year + housing_fee + 
                scholarship + price + commute + privacy + study_env + 
                social + facilities + rules,
              data = train_data, family = binomial) # 2nd model with all IVs excluding faculty (too many categories)
summary(logit2)
test_data$pred2 <- predict(logit2, newdata = test_data, type = "response")
roc2 <- roc(test_data$dorm, test_data$pred2)
auc(roc2)
# AIC=169.43, AUC=0.665 - nearly as good as logit1 (AUC 0.6875) despite dropping the
# 9-level faculty factor, and with far fewer parameters. This is a better balance of fit vs.
# simplicity than logit1.

logit2_sustain <- glm(dorm ~ student_status + year + housing_fee +
                scholarship + price + commute + privacy + study_env +
                social + facilities + rules + sustainability + green_premium,
              data = train_data, family = binomial)
summary(logit2_sustain)
test_data$pred2_sustain <- predict(logit2_sustain, newdata = test_data, type = "response")
roc2_sustain <- roc(test_data$dorm, test_data$pred2_sustain)
auc(roc2_sustain)  # compare against auc(roc2) to see if the new questions add predictive value
# Again, sustainability and green_premium are not significant and don't improve AUC over logit2.


logit_simple <- glm(dorm ~ price + year,
                    data = train_data,
                    family = binomial)# 3rd Model kept only significant IVs: price and year
summary(logit_simple)

pred_simple <- predict(logit_simple, newdata = test_data, type = "response")# Test performance on test set
roc_simple <- roc(test_data$dorm, pred_simple)
auc(roc_simple)
table(test_data$dorm, pred_simple > 0.5)#Confusion matrix
#overall Accuracy = (8+16)/40 = 60%; Sensitivity(TPR, dorm) = 16/20 = 80% (Good at identifying dorm students);
#Specificity(TNR, off-campus) = 8/20 = 40% (Poor at identifying off-campus). AIC=163.18 is the LOWEST
#(best) of any model tried, but AUC drops to 0.555 - barely above chance. This is the key lesson from
#re-running on the larger sample: the lowest-AIC model is an IN-SAMPLE fit measure, and does not
#guarantee the best OUT-OF-SAMPLE discrimination (AUC). logit2 (AUC 0.665, AIC 169.43) generalizes
#much better despite a slightly higher AIC, and should replace logit_simple as our preferred model.

# Tried adding an interaction term (like in lecture), but didn't get a better model
logit4 <- glm(dorm ~ student_status * scholarship + social + commute,
              data = train_data, family = binomial)
summary(logit4)
test_data$pred_prob4 <- predict(logit4, newdata = test_data, type = "response")
roc4 <- roc(test_data$dorm, test_data$pred_prob4)
auc(roc4)

# FINAL MODEL FOR REPORT
summary(logit2)  # Predict housing preference using all IVs except faculty (best out-of-sample AUC
                 # among interpretable models: 0.665, vs. logit_simple's 0.555)

# Model interpretation (updated on the larger, sustainability-augmented dataset, n=160):
# 1. AIC = 169.43, AUC = 0.665 - best generalization of the interpretable models tried
# 2. year4 remains the strongest individual predictor (large negative coefficient, consistent with logit1)
# 3. Adding sustainability + green_premium (logit2_sustain) leaves AUC unchanged at 0.665 while
#    making AIC worse (173.4 vs 169.43) - confirms these two variables add no predictive value
# 4. logit_simple (price + year only) has the best AIC (163.18) but the worst AUC (0.555) of the
#    serious candidates - a reminder that AIC and out-of-sample AUC can disagree, and AUC is what
#    matters for a model meant to predict, not just fit, on new students

#ACKNOWLEDGE SHORTCOMINGS & LIMITATIONS
#1. Sample size: N=160 (up from 117 in the previous round; still modest for 12+ predictors)
#2. Model AUC: best model (logit2) reaches 0.665, still short of the 0.7 "good prediction" threshold
#3. Missing predictors: family income, distance from home, accommodation quality
#4. Class balance: now roughly even (81 dorm / 81 non-dorm), unlike the previous round's imbalance -
#   this appears to be a major reason AUC improved so much from the last run
#5. Only a handful of predictors reach significance in any model: housing choice may be more complex
#   than these self-reported ratings can capture
#6. Single train/test split: results may still vary with different splits - cross-validation (see
#   analysis/python_reanalysis_honest_cv.py in the repo) shows RF AUC ranging 0.68-0.84 across folds
#7. Sustainability features and willingness to pay a green premium, now measured for the first time,
#   are not significant in any logistic model and worsen AIC when added - consistent with the
#   CART/Random Forest finding that they are not (yet) meaningful drivers of housing choice

#RECOMMENDATIONS FOR FUTURE RESEARCH
#1. Collect an even larger sample (target 300+) to stabilize estimates further
#2. Add socioeconomic factors (family income, financial independence)
#3. Include distance/location data (proximity to home)
#4. Use k-fold cross-validation for robust evaluation (already partly addressed in analysis/)
#5. Consider non-linear models or machine learning approaches (see 02_cart_random_forest.R)
#6. Re-test sustainability preferences in a future round as environmental features gain visibility
