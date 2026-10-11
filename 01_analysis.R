# ============================================================
# Mobile Usage Pattern Analysis Using R
# Script 1: Cleaning, EDA, Clustering, Modelling
# Dataset: "Mobile Device Usage and User Behavior Dataset" (Kaggle)
# Put the file at: data/user_behavior_dataset.csv
# ============================================================
pkgs <- c("dplyr","ggplot2","caret","randomForest","rpart","corrplot")
for (p in pkgs) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
invisible(lapply(pkgs, library, character.only = TRUE))
set.seed(123)
dir.create("outputs", showWarnings = FALSE); dir.create("models", showWarnings = FALSE)

# ---------- Stage 1: Load data ----------
raw <- read.csv("data/user_behavior_dataset.csv", stringsAsFactors = FALSE)
stopifnot(ncol(raw) == 11)
names(raw) <- c("UserID","DeviceModel","OS","AppUsageMin","ScreenOnHrs",
                "BatteryDrain","AppsInstalled","DataUsageMB","Age","Gender","BehaviorClass")
cat("Dimensions:", dim(raw), "\n"); str(raw); print(summary(raw))

# ---------- Stage 2: Data cleaning ----------
cat("\nMissing values per column:\n"); print(colSums(is.na(raw)))
cat("Duplicate rows:", sum(duplicated(raw)), "\n")
df <- raw %>% distinct() %>% na.omit()
df$DeviceModel   <- factor(df$DeviceModel)
df$OS            <- factor(df$OS)
df$Gender        <- factor(df$Gender)
df$BehaviorClass <- factor(paste0("C", df$BehaviorClass), levels = paste0("C", 1:5))
saveRDS(df, "models/clean_data.rds")

# ---------- Stage 3: Visualization ----------
p1 <- ggplot(df, aes(AppUsageMin)) + geom_histogram(bins = 30, fill = "steelblue") +
  ggtitle("Distribution of Daily App Usage Time (min)")
ggsave("outputs/01_app_usage_hist.png", p1, width = 7, height = 4)

p2 <- ggplot(df, aes(BehaviorClass, ScreenOnHrs, fill = BehaviorClass)) + geom_boxplot() +
  ggtitle("Screen-On Time by User Behavior Class") + theme(legend.position = "none")
ggsave("outputs/02_screen_by_class.png", p2, width = 7, height = 4)

p3 <- ggplot(df, aes(OS, fill = Gender)) + geom_bar(position = "dodge") +
  ggtitle("OS Usage by Gender")
ggsave("outputs/03_os_gender.png", p3, width = 7, height = 4)

p4 <- ggplot(df, aes(ScreenOnHrs, BatteryDrain, colour = BehaviorClass)) +
  geom_point(alpha = .6) + ggtitle("Screen-On Time vs Battery Drain")
ggsave("outputs/04_screen_vs_battery.png", p4, width = 7, height = 4)

num_cols <- c("AppUsageMin","ScreenOnHrs","BatteryDrain","AppsInstalled","DataUsageMB","Age")
png("outputs/05_correlation.png", 800, 700)
corrplot(cor(df[, num_cols]), method = "color", addCoef.col = "black", tl.cex = .8)
dev.off()

long <- stack(scale(df[, num_cols]) |> as.data.frame())
p6 <- ggplot(long, aes(ind, values)) + geom_boxplot(fill = "orange") +
  ggtitle("Boxplots (scaled) for outlier check") + xlab("")
ggsave("outputs/06_boxplots.png", p6, width = 8, height = 4)

# ---------- Stage 4: Usage pattern clustering (K-Means) ----------
sc <- scale(df[, c("AppUsageMin","ScreenOnHrs","BatteryDrain","DataUsageMB","AppsInstalled")])
wss <- sapply(1:8, function(k) kmeans(sc, k, nstart = 20)$tot.withinss)
png("outputs/07_elbow.png", 700, 450)
plot(1:8, wss, type = "b", xlab = "k", ylab = "Total within-SS", main = "Elbow Method"); dev.off()
km <- kmeans(sc, centers = 3, nstart = 25)
df$Cluster <- factor(km$cluster)
print(df %>% group_by(Cluster) %>%
        summarise(across(all_of(c("AppUsageMin","ScreenOnHrs","BatteryDrain","DataUsageMB","AppsInstalled")), mean),
                  n = n()))

# ---------- Stage 5: Train/test split (80/20) ----------
idx   <- createDataPartition(df$BehaviorClass, p = 0.8, list = FALSE)
train <- df[idx, ]; test <- df[-idx, ]
cat("Train rows:", nrow(train), "| Test rows:", nrow(test), "\n")
feat <- c("OS","AppUsageMin","ScreenOnHrs","BatteryDrain","AppsInstalled","DataUsageMB","Age","Gender")

# ---------- Stage 6: Models ----------
# (a) Classification of BehaviorClass : Decision Tree + Random Forest
dt <- rpart(BehaviorClass ~ ., data = train[, c("BehaviorClass", feat)], method = "class")
dt_pred <- predict(dt, test, type = "class")
rf <- randomForest(BehaviorClass ~ ., data = train[, c("BehaviorClass", feat)],
                   ntree = 300, importance = TRUE)
rf_pred <- predict(rf, test)
cm_dt <- confusionMatrix(dt_pred, test$BehaviorClass)
cm_rf <- confusionMatrix(rf_pred, test$BehaviorClass)
print(cm_dt); print(cm_rf)

# (b) Regression of BatteryDrain : Linear Regression + Random Forest
reg_feat <- c("AppUsageMin","ScreenOnHrs","AppsInstalled","DataUsageMB","Age","OS","Gender")
lm_m  <- lm(BatteryDrain ~ ., data = train[, c("BatteryDrain", reg_feat)])
rf_r  <- randomForest(BatteryDrain ~ ., data = train[, c("BatteryDrain", reg_feat)], ntree = 300)
rmse  <- function(a, b) sqrt(mean((a - b)^2))
r2    <- function(a, b) cor(a, b)^2
lm_p  <- predict(lm_m, test); rfr_p <- predict(rf_r, test)
print(summary(lm_m))

pa <- ggplot(data.frame(Obs = test$BatteryDrain, Pred = lm_p), aes(Obs, Pred)) +
  geom_point(alpha = .5) + geom_abline(colour = "red") + ggtitle("Linear Regression: Predicted vs Actual")
ggsave("outputs/08_lm_pred_vs_actual.png", pa, width = 6, height = 5)
pb <- ggplot(data.frame(Obs = test$BatteryDrain, Pred = rfr_p), aes(Obs, Pred)) +
  geom_point(alpha = .5) + geom_abline(colour = "red") + ggtitle("Random Forest: Predicted vs Actual")
ggsave("outputs/09_rf_pred_vs_actual.png", pb, width = 6, height = 5)

png("outputs/10_rf_importance.png", 800, 500); varImpPlot(rf, main = "RF Variable Importance (BehaviorClass)"); dev.off()

# ---------- Stage 7: Comparison ----------
comp <- data.frame(
  Task   = c("Classification","Classification","Regression","Regression"),
  Model  = c("Decision Tree","Random Forest","Linear Regression","Random Forest Regression"),
  Accuracy = c(cm_dt$overall["Accuracy"], cm_rf$overall["Accuracy"], NA, NA),
  RMSE   = c(NA, NA, rmse(lm_p, test$BatteryDrain), rmse(rfr_p, test$BatteryDrain)),
  R2     = c(NA, NA, r2(lm_p, test$BatteryDrain), r2(rfr_p, test$BatteryDrain)))
print(comp); write.csv(comp, "outputs/model_comparison.csv", row.names = FALSE)

# ---------- Stage 8: Save model for Shiny ----------
saveRDS(rf, "models/rf_behavior.rds")
saveRDS(df, "models/clean_data.rds")
cat("\nDone. Plots in outputs/, models in models/\n")
cat("NOTE: if accuracy is ~1.0, discuss possible synthetic data / leakage in your report.\n")
