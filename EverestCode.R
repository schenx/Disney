##
##  
##  Expedition Everest Wait Time project Code
##  Steven Chen - June 25 2019
##  for EdX HarvardX PH125.9x Data Science: Capstone
##
##  Code to load, analyze, and model Wait Time data for the Walt Disney World
##  Animal Kingdom attraction Expedition Everest
##
##

# Load needed packages
library(dslabs)
library(dplyr)
library(tidyverse)
library(caret)
library(lubridate)
library(httr)

# Load Everest Data file
# Data files are stored locally
# lubridate used to make date and time data uniform

EverestData <- read.csv("https://raw.githubusercontent.com/schenx/Disney/master/Everest_Data_Small.csv")
EverestData$DATE <- lubridate::mdy(EverestData$DATE)
EverestData$Time <- lubridate::hm(EverestData$Time)

# Create a "TimeHour" field which converts time of day from HH:MM to hours in decimals
EverestData <- EverestData %>% mutate(TimeHour = (as.numeric(EverestData$Time))/3600) 

# Summary Statistics
ev_mean <- mean(EverestData$Wait_Time)
Summary_Stats <- data_frame( Stat = "Mean", Value = ev_mean)

ev_median <- median(EverestData$Wait_Time)
Summary_Stats <- bind_rows(Summary_Stats, data_frame(Stat = "Median", Value = ev_median))

ev_sd <- sd(EverestData$Wait_Time)
Summary_Stats <- bind_rows(Summary_Stats, data_frame(Stat = "SD", Value = ev_sd))

ev_min <- min(EverestData$Wait_Time)
Summary_Stats <- bind_rows(Summary_Stats, data_frame(Stat = "Min", Value = ev_min))

ev_max <- max(EverestData$Wait_Time)
Summary_Stats <- bind_rows(Summary_Stats, data_frame(Stat = "Max", Value = ev_max))

Summary_Stats

ev_quantile <- quantile(EverestData$Wait_Time)
ev_quantile

# Histogram of Wait Times
ggplot(EverestData, aes(Wait_Time)) + geom_histogram(binwidth=5, fill="tomato3") +
  labs(title="Expedition Everest Wait Time Histogram (binwidth = 5min)")


# Average Wait Time by Month of Year
Month_Avg <- EverestData %>% group_by(MONTHOFYEAR) %>% summarize(Avg_Wait = mean(Wait_Time))
ggplot(Month_Avg, aes(x=MONTHOFYEAR, y=Avg_Wait)) + geom_bar(stat="identity", width=.5, fill="tomato3") +
  labs(title="Expedition Everest Average Wait Time by Month")
  
# Plot Daily Average Wait Time by Date
# Find Daily Averages and Sort
Day_Avg <- EverestData %>% group_by(DATE) %>% summarize(Day_Avg_Wait = mean(Wait_Time)) 
dplyr::arrange(Day_Avg, DATE)
# Plot Daily Averages
ggplot(Day_Avg, aes(x=DATE, y=Day_Avg_Wait)) + geom_line() + geom_smooth(method = "lm", color="red") +
  labs(title = "Expedition Everest Daily Average Wait Time by Date")

# Average Wait Time by Day of Week
DoW_Avg <- EverestData %>% group_by(DAYOFWEEK) %>% summarize(Avg_Wait = mean(Wait_Time))
ggplot(DoW_Avg, aes(x=DAYOFWEEK, y=Avg_Wait)) + geom_bar(stat="identity", width=.5, fill="tomato3") +
  labs(title="Expedition Everest Average Wait Time by Day of Week")

# Wait Time during Time of Day
ggplot(EverestData, aes(x=TimeHour, y=Wait_Time)) + geom_point(color="tomato3") +
  labs(title = "Expedition Everest Wait Time During Time of Day (by Hour)")

# Create Validation and Test Sets
set.seed(3) 
test_index <- createDataPartition(y = EverestData$Wait_Time, times = 1, p = 0.1, list = FALSE)
Everest_Train <- EverestData[-test_index,]
validation <- EverestData[test_index,]
rm(test_index)

# Code the Root Mean Square Error function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Naive Approach
mu_hat <- mean(EverestData$Wait_Time)
naive_rmse <- RMSE(validation$Wait_Time, mu_hat)
rmse_results <- data_frame(method = "Naive Approach", RMSE = naive_rmse)
rmse_results

# Linear Model Approach
fit <- lm(Wait_Time ~ DAYOFWEEK + DAYOFYEAR + WEEKOFYEAR + MONTHOFYEAR
          + AKEMHMORN + AKEMHEVE + AKHOURSEMH + AKHOURS + TimeHour, data = Everest_Train)
y_hat <- predict(fit, validation)
linear_rmse <- RMSE(validation$Wait_Time, y_hat)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Linear Model Approach", RMSE = linear_rmse))
rmse_results

# K Nearest Neighbors Approach
train_knn <- train(Wait_Time ~., method="knn", data=Everest_Train)
y_hat_knn <- predict(train_knn, validation, type="raw")
knn_rmse <- RMSE(validation$Wait_Time, y_hat_knn)
rmse_results <- bind_rows(rmse_results, data_frame(method = "K Nearest Neighbor Approach",
                                                   RMSE = knn_rmse))
rmse_results


# Regression Trees Approach
train_rpart <- train(Wait_Time ~., method="rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)),data=Everest_Train)
y_hat_rpart <- predict(train_rpart, validation, type="raw")
rpart_rmse <- RMSE(validation$Wait_Time, y_hat_rpart)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Regression Trees Approach",
                                                 RMSE = rpart_rmse))
rmse_results


# Random Forest Approach
library(randomForest)

train_rf <- randomForest(Wait_Time ~., data=Everest_Train)
y_hat_rf <- predict(train_rf, validation, type="response")
rf_rmse <- RMSE(validation$Wait_Time, y_hat_rpart)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Random Forest Approach",
                                                   RMSE = rpart_rmse))
rmse_results


# Ensemble (KNN and Random Forest) Approach
y_hat <- (y_hat_rf + y_hat_knn)/2
ensemble_rmse <- RMSE(validation$Wait_Time, y_hat)

rmse_results <- bind_rows(rmse_results, data_frame(method = "Ensemble (KNN + Random Forest) Approach",
                                                  RMSE = ensemble_rmse))
rmse_results

# Variable Importance


