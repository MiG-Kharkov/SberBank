library(data.table)
library(Matrix)
library(xgboost)
library(caret)
library(dplyr)

# Load CSV files
cat("Read data")
data.train <- fread('Data/train.csv', na.strings = "NA")
data.test <- fread("Data/test.csv", sep=",", na.strings = "NA")
data.macro = fread("Data/macro.csv", sep=",", na.strings="NA")

# resample train data
data.train = data.train[price_doc>1000000 & price_doc<111111112,]

# re_investment = 
#   data.train %>% 
#   filter(product_type=='Investment',timestamp>='2011-10-01') %>% 
#   group_by(ts=substring(timestamp,1,7)) %>% 
#   summarise(n=n(),
#             n1M=sum(ifelse(price_doc<=1000000,1,0))/n(),
#             n2M=sum(ifelse(price_doc==2000000,1,0))/n(),
#             n3M=sum(ifelse(price_doc==3000000,1,0))/n())
# 
# m1=floor(mean(re_investment$n1M[re_investment$ts>='2015-01'])/10*nrow(data.train)) #undersampling by magic numbers
# m2=floor(mean(re_investment$n2M[re_investment$ts>='2015-01'])/3*nrow(data.train)) #undersampling by magic numbers
# m3=floor(mean(re_investment$n3M[re_investment$ts>='2015-01'])/2*nrow(data.train)) 
# 
# set.seed(1)
# i1 = data.train %>% filter(price_doc<=1000000,product_type=='Investment') %>% sample_n(m1)
# i2 = data.train %>% filter(price_doc==2000000,product_type=='Investment') %>% sample_n(m2)
# i3 = data.train %>% filter(price_doc==3000000,product_type=='Investment') %>% sample_n(m3)
# 
# data.train = data.train %>% filter(!(price_doc<=1000000 & product_type=='Investment'))
# data.train = data.train %>% filter(!(price_doc==2000000 & product_type=='Investment'))
# data.train = data.train %>% filter(!(price_doc==3000000 & product_type=='Investment'))
# 
# data.train = rbind(data.train,i1,i2,i3) %>% arrange(id)
# data.train = as.data.table(data.train)

# Transform target variable, so that we can use RMSE in XGBoost
data.train[,price_doc := log1p(as.integer(price_doc))]

data.train[is.na(data.train)] = -1
data.test[is.na(data.test)] = -1

# Combine data.tables
data <- rbind(data.train, data.test,fill=TRUE)

#clean full_sq and life_sq. sometime full_sq is smaller than life_sq
data[ ,life_sq:=ifelse(is.na(life_sq),full_sq,life_sq)]
data[ ,full_sq:=ifelse(life_sq>full_sq,life_sq,full_sq)]

#build_year
data[ ,build_year:=ifelse((build_year >1690 & build_year<2020),build_year,'NA')]
data[ ,build_year:=as.integer(build_year)]

#num_rooms
data[ ,num_room:=ifelse(num_room==0,'NA',num_room)]
data[ ,num_room:=as.integer(num_room)]

#state
data[ ,state := ifelse(state==33,3,state)]
#data[, state:= as.integer(state)]


# Convert characters to factors/numeric
cat("Feature engineering")
data[,":="(thermal_power_plant_raion=ifelse(thermal_power_plant_raion=="no",0,1)
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
data[,":="(#date_yday=yday(timestamp)
  # date_month=month(timestamp)
  date_year=year(timestamp)
  #,date_week=week(timestamp)
  #,date_mday=mday(timestamp)
  #,date_wday=wday(timestamp)
)]

#data[,":="(date_yearmonth=date_year*100+date_month
#         ,date_yearweek=date_year*100+date_week)]

#data[,date_yearmonth_count := .N, keyby = date_yearmonth]
#data[,date_yearweek_count := .N, keyby = date_yearweek]

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

#data[, date_year:=as.factor(date_year)]

#macro data
data.macro = data.macro[,.(timestamp,
                           balance_trade,
                           balance_trade_growth,
                           eurrub,
                           average_provision_of_build_contract,
                           micex_rgbi_tr,
                           micex_cbi_tr, 
                           deposits_rate, 
                           mortgage_value, mortgage_rate,
                           income_per_cap, 
                           museum_visitis_per_100_cap,
                           cpi,
                           apartment_build)]


features = colnames(data)
for (f in features){
  if( (class(data[[f]]) == "character") || (class(data[[f]]) == "factor"))
  {
    levels = unique(data[[f]])
    data[[f]] = as.numeric(factor(data[[f]], level = levels)) 
  }
}

data[, material:=as.factor(material)]
data[, build_year := as.factor(build_year)]
data[, state:=as.factor(state)]
data[, product_type:=as.factor(product_type)]
data[, sub_area:=as.factor(sub_area)]
data[, ecology:=as.factor(ecology)]
#data[, date_week:=as.factor(date_week)]
#data[, date_month:=as.factor(date_month)]


# one-hot-encoding features
data = as.data.frame(data)
ohe_feats = c('material', 'build_year', 'state', 'product_type', 'sub_area', 'ecology')
dummies = dummyVars(~ material + build_year + state + product_type + sub_area + ecology , data = data)
df_all_ohe <- as.data.frame(predict(dummies, newdata = data))
df_all_combined <- cbind(data[,-c(which(colnames(data) %in% ohe_feats))],df_all_ohe)
data = as.data.table(df_all_combined)

#data = data_v04
data[,c("date_year", "timestamp", "young_male", "school_education_centers_top_20_raion", "0_17_female", "railroad_1line", "7_14_female", "0_17_all", "children_school",
        "16_29_male", "mosque_count_3000", "female_f", "church_count_1000", "railroad_terminal_raion",
        "mosque_count_5000", "big_road1_1line", "mosque_count_1000", "7_14_male", "0_6_female", "oil_chemistry_raion",
        "young_all", "0_17_male", "ID_bus_terminal", "university_top_20_raion", "mosque_count_500","ID_big_road1",
        "ID_railroad_terminal", "ID_railroad_station_walk", "ID_big_road2", "ID_metro", "ID_railroad_station_avto",
        "0_13_all", "mosque_count_2000", "work_male", "16_29_all", "young_female", "work_female", "0_13_female",
        "ekder_female", "7_14_all", "big_church_count_500",
        "leisure_count_500", "cafe_sum_1500_max_price_avg", "leisure_count_2000",
        "office_count_500", "male_f", "nuclear_reactor_raion", "0_6_male", "church_count_500", "build_count_before_1920",
        "thermal_power_plant_raion", "cafe_count_2000_na_price", "cafe_count_500_price_high",
        "market_count_2000", "museum_visitis_per_100_cap", "trc_count_500", "market_count_1000", "work_all", "additional_education_raion",
        "build_count_slag", "leisure_count_1000", "0_13_male", "office_raion",
        "raion_build_count_with_builddate_info", "market_count_3000", "ekder_all", "trc_count_1000", "build_count_1946-1970",
        "office_count_1500", "cafe_count_1500_na_price", "big_church_count_5000", "big_church_count_1000", "build_count_foam",
        "church_count_1500", "church_count_3000", "leisure_count_1500",
        "16_29_female", "build_count_after_1995", "cafe_avg_price_1500", "office_sqm_1000", "cafe_avg_price_5000", "cafe_avg_price_2000",
        "big_church_count_1500", "full_all", "cafe_sum_5000_min_price_avg",
        "office_sqm_2000", "church_count_5000","0_6_all", "detention_facility_raion", "cafe_avg_price_3000")
     :=NULL]

varnames <- setdiff(colnames(data), c("id","price_doc"))

cat("Create sparse matrix")
# To sparse matrix
train_sparse <- Matrix(as.matrix(sapply(data[price_doc > -1,varnames,with=FALSE],as.numeric)), sparse=TRUE)
test_sparse <- Matrix(as.matrix(sapply(data[is.na(price_doc),varnames,with=FALSE],as.numeric)), sparse=TRUE)
y_train <- data[!is.na(price_doc),price_doc]
test_ids <- data[is.na(price_doc),id]

dtrain <- xgb.DMatrix(data=train_sparse, label=y_train)
dtest <- xgb.DMatrix(data=test_sparse);

gc()

# Params for xgboost
param <- list(objective="reg:linear",
              eval_metric = "rmse",
              booster = "gbtree",
              eta = .05,
              gamma = 1,
              max_depth = 4,
              min_child_weight = 1,
              subsample = .7,
              colsample_bytree = .7
)


# cvFoldsList <- createFolds(1:nrow(data.train), k = 5)
# 
# xgb_cv <- xgb.cv(data = dtrain,
#                 params = param,
#                 nrounds = 1500,
#                 maximize = FALSE,
#                 prediction = TRUE,
#                 folds = cvFoldsList,
#                 print.every.n = 5,
#                 early.stop.round = 100
# );
# gc()
# rounds <- which.min(xgb_cv$dt[, test.rmse.mean])
rounds = 5#501
mpreds = data.table(id=test_ids)

for(random.seed.num in 1:2) {
  print(paste("[", random.seed.num , "] training xgboost begin ",sep=""," : ",Sys.time())) 
  set.seed(random.seed.num)
  xgb_model <- xgb.train(data = dtrain,
                         params = param,
                         watchlist = list(train = dtrain),
                         nrounds = rounds,
                         verbose = 1,
                         print.every.n = 5
                         # missing='NAN'
  )
  
  vpreds = predict(xgb_model,dtest) 
  mpreds = cbind(mpreds, vpreds)    
  colnames(mpreds)[random.seed.num+1] = paste("pred_seed_", random.seed.num, sep="")
}

# Feature importance
#cat("Plotting feature importance")
#names <- dimnames(train_sparse)[[2]]
#importance_matrix <- xgb.importance(names,model=xgb_model)
#xgb.plot.importance(importance_matrix[1:20,])

mpreds_2 = mpreds[, id:= NULL]
mpreds_2 = mpreds_2[, price_doc := rowMeans(.SD)]
mpreds_2[, ':='(id = data.test$id, timestamp=data.test$timestamp)]
submission = data.table(id=test_ids, price_doc=expm1(mpreds$price_doc))

write.table(submission, "sberbank_submission_v05.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)
