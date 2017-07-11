library(data.table)
library(Matrix)
library(xgboost)
library(caret)
library(gbm)
library(glmnet)

# Load CSV files
cat("Read data")
data.train <- fread("Data/train.csv", sep=",", na.strings = "NA")
data.test <- fread("Data/test.csv", sep=",", na.strings = "NA")

# Transform target variable, so that we can use RMSE in XGBoost
data.train[,price_doc := log1p(as.integer(price_doc))]
data.test[,price_doc := "-1"]

# Combine data.tables
data <- rbind(data.train, data.test)


# Convert characters to factors/numeric
cat("Feature engineering")
data[,":="(product_type=as.factor(product_type)
           ,sub_area=as.factor(sub_area)
           ,ecology=as.factor(ecology)
           ,thermal_power_plant_raion=ifelse(thermal_power_plant_raion=="no",0,1)
           ,incineration_raion=ifelse(incineration_raion=="no",0,1)
           ,oil_chemistry_raion=ifelse(oil_chemistry_raion=="no",0,1)
           ,radiation_raion=ifelse(radiation_raion=="no",0,1)
           ,railroad_terminal_raion=ifelse(railroad_terminal_raion=="no",0,1)
           ,big_market_raion=ifelse(big_market_raion=="no",0,1)
           ,nuclear_reactor_raion=ifelse(nuclear_reactor_raion=="no",0,1)
           ,detention_facility_raion=ifelse(detention_facility_raion=="no",0,1)
           ,culture_objects_top_25=ifelse(culture_objects_top_25=="no",0,1)
           ,water_1line=ifelse(water_1line=="no",0,1)
           ,big_road1_1line=ifelse(big_road1_1line=="no",0,1)
           ,railroad_1line=ifelse(railroad_1line=="no",0,1)
)]

# Date features
data[,timestamp := as.Date(timestamp)]
data[,":="(date_yday=yday(timestamp)
           ,date_month=month(timestamp)
           ,date_year=year(timestamp)
           ,date_week=week(timestamp)
           ,date_mday=mday(timestamp)
           ,date_wday=wday(timestamp)
)]
data[,":="(date_yearmonth=date_year*100+date_month
           ,date_yearweek=date_year*100+date_week
)]
data[,date_yearmonth_count := .N, keyby = date_yearmonth]
data[,date_yearweek_count := .N, keyby = date_yearweek]

# Count NA's
data[,na_count := rowSums(is.na(data))]

# Some relative features
data[,":="(rel_floor = floor/max_floor
           ,diff_floor = max_floor-floor
           ,rel_kitchen_sq = kitch_sq/full_sq
           ,rel_life_sq = life_sq/full_sq
           ,rel_kitchen_life = kitch_sq/life_sq
           ,rel_sq_per_floor = full_sq/floor
           ,diff_life_sq = full_sq-life_sq
           ,building_age = date_year - build_year
)]


# Load macro data
cat("Load macro data")
data.macro <- fread("Data/macro.csv", sep=",", na.strings = "NA")
data.macro[,timestamp := as.Date(timestamp)]
data.macro <- sapply(data.macro,as.numeric) # Simply cast to numeric here, we can do feature engineering later
data <- merge(data, data.macro, by="timestamp", all.x=TRUE)
rm(data.macro, data.train)
gc()

# To sparse matrix
cat("Create sparse matrix")
varnames <- setdiff(colnames(data), c("id", "price_doc", "timestamp"))
my.RMSE = function(model, x, y, round = T, ...){
  pred_valid = predict(model, x, ...)
  if(isTRUE(round))
    pred_valid  = log(round(exp(pred_valid)/100, digits = 0)*100)
  sqrt(mean((pred_valid - y)**2))
}

set.seed(111)
idx = createDataPartition(1:30471, p = .85, list = F)

train.x = as.data.frame(sapply(data[price_doc > -1, varnames, with = FALSE],as.numeric))
train.y = as.data.frame(sapply(data[price_doc > -1, price_doc],as.numeric))
names(train.y) <- "logSalePrice"

train_sub.x = train.x[idx,]
train_sub.y = train.y[idx,]

valid.x = train.x[-idx,]
valid.y = train.y[-idx,]

params <- list(
  "objective"           = "reg:linear",
  "eval_metric"         = "rmse",
  # "seed"                = 123,
  "eta"                 = 0.2,
  "max_depth"           = 10,
  # "min_child_weight"    = 10,
  "gamma"               = 0.01,
  "subsample"           = 0.9,
  # "colsample_bytree"    = 0.95,
  "alpha"               = 0.1,
  "lambda"              = 10
)

X <- xgb.DMatrix(as.matrix(train_sub.x), label = train_sub.y)
set.seed(123)
resXGB <- xgboost(data = X, params = params, nrounds = 200) #was 60

my.RMSE(resXGB, as.matrix(train_sub.x),train_sub.y)
my.RMSE(resXGB, as.matrix(valid.x), valid.y)

importance <- xgb.importance(colnames(X), model = resXGB)
# install.packages("Ckmeans.1d.dp")
xgb.ggplot.importance(importance[1:20,])

# Predict and output csv
X <- xgb.DMatrix(as.matrix(train.x), label = train.y$logSalePrice)
set.seed(123)

resXGB <- xgboost(data = X, params = params, nrounds = 100) #was 60

# make result based only on xgb
test.x = as.matrix(as.data.frame(sapply(data[price_doc == -1, varnames, with = FALSE],as.numeric)))
pred_xgb <- predict(resXGB, test.x)
pred_xgb <- exp(pred_xgb)
res = data.table(Id = data[price_doc == -1]$id, price_doc = pred_xgb)
setnames(res, c("id", "price_doc"))

write.csv(res, "submission1.csv" ,row.names = F)
