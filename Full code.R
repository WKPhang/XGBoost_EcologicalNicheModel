# Load library
library(dplyr)
library(maptools)
library(raster)
library(usdm)
library(xgboost)
library(caret)
library(pROC)
library(MLmetrics)
library(SHAPforxgboost)

#### PART 1: Input occurence and background data from Maxent output folder
# NOTE: This script only works if the Maxent software was run with replicated run type of either Crossvalidate or Subsample
# The predicted value and partitioning of occurrence data are created with default name pattern "samplePredictions.csv"

filenames <- list.files(path="MAXENT OUTPUT FILE LOCATION", 
                        pattern="samplePredictions.csv", full.names=T)

setwd("MAXENT OUTPUT FILE LOCATION") 

for (file in filenames[1:nrow(filenames)]){
  # if the merged dataset exist, append to it
  if (exists("dataset")){
    temp_dataset <-read.table(file, header=T, sep=",")
    dataset<-left_join(dataset, temp_dataset[,c(1:3)], by = c('X' = 'X','Y'='Y'))
    column_name <- tools::file_path_sans_ext(basename(file))
    column_name <- sub("_samplePredictions", "", column_name)
    column_name <- sub("species", "train", column_name)
    colnames(dataset)[ncol(dataset)] <- column_name
    rm(temp_dataset, column_name)
  }
  # if the merged dataset does not exist, create it
  if (!exists("dataset")){
    temp_dataset <- read.table(file, header=T, sep=",")
    dataset <- temp_dataset[,c(1,2,3)]
    column_name <- tools::file_path_sans_ext(basename(file))
    column_name <- sub("_samplePredictions", "", column_name)
    column_name <- sub("species", "train", column_name)
    colnames(dataset)[ncol(dataset)] <- column_name
    rm(temp_dataset, column_name)
  }
}
dataset <- dataset %>% 
  mutate_at(vars(3:ncol(dataset)),
            ~as.numeric(recode(.,
                               'train'= 1,
                               'test'= 0)))
dataset

# Input background data
filenames_bg <- list.files(path="MAXENT OUTPUT FILE LOCATION",
                           pattern="backgroundPredictions.csv", full.names=T)
filenames_bg

for (file in filenames_bg[1:nrow(filenames_bg)]){
  # if the merged dataset exist, append to it
  if (exists("dataset_bg")){
    temp_dataset <-read.table(file, header=T, sep=",")
    dataset_bg<-cbind(dataset_bg, temp_dataset[,c(1:2)])
    column_name <- tools::file_path_sans_ext(basename(file))
    column_name <- sub("Pk", "", column_name)
    colnames(dataset_bg)[ncol(dataset_bg)-1] <- sub("_backgroundPredictions","_X", column_name)
    colnames(dataset_bg)[ncol(dataset_bg)] <- sub("_backgroundPredictions","_Y", column_name)
    rm(temp_dataset, column_name)
  }
  # if the merged dataset does not exist, create it
  if (!exists("dataset_bg")){
    temp_dataset <- read.table(file, header=T, sep=",")
    dataset_bg <- temp_dataset[,c(1:2)]
    column_name <- tools::file_path_sans_ext(basename(file))
    column_name <- sub("Pk", "", column_name)
    colnames(dataset_bg)[ncol(dataset_bg)-1] <- sub("_backgroundPredictions","_X", column_name)
    colnames(dataset_bg)[ncol(dataset_bg)] <- sub("_backgroundPredictions","_Y", column_name)
    rm(temp_dataset, column_name)
  }
}
dataset_bg

rm(file, filenames, filenames_bg)

#### PART 2: Input covariate data 
# NOTE: Ensure that all raster layers are in ascii format

# Load raster (covariates of interest) files
grids <- list.files(path="COVARIATE RASTER FOLDER LOCATION",
                    pattern = "*.asc$")
grids

setwd("COVARIATES RASTER FOLDER LOCATION")
s <- stack(paste0(grids))

# Create overlapping mask
res <- raster(s)
overlap_mask <- any(s, na.rm=F)

# Create dMatrix from raster data
matrix_s <- as.matrix(s)
dmatrix_s <- xgb.DMatrix(matrix_s[,-15])

#### PART 3: Set up XGBoost parameter
xgb_control_repcv <- trainControl(
  method = "repeatedcv", 
  number = 5,
  repeats = 1,
  allowParallel = TRUE, 
  verboseIter = F, 
  returnData = FALSE,
  summaryFunction=twoClassSummary,
  classProbs=T,
  returnResamp = "final"
)

xgb_grid <- base::expand.grid(
  list(
    nrounds = seq(500,500, by=100),
    max_depth = seq(5,5, by=2), 
    colsample_bytree = seq(0.5,0.5,by=0.1), 
    eta = seq(0.02,0.02, by=0.01),
    gamma = seq(0,0,by=1),
    min_child_weight = seq(1,1, by=1),  
    subsample = seq(0.7,0.7, by=0.1))
)

#### PART 4: Loop of full XGBoost model run and ensemble modelling
# The loop starts at line 126 
for(dtset_i in 1:30){
print(paste("start ", dtset_i, sep=""))
{
  case_presence <- dataset[,c(1:2,dtset_i+2,dtset_i+2)]
  background <- dataset_bg[,c((dtset_i*2)-1, dtset_i*2)]
  colnames(case_presence)[1:4] <- c("X", "Y", "train","test_0")
  colnames(background)[1:2] <- c("X","Y")
  background$train <- 1
  background$test_0 <- 0
  case_presence$presence <- 1
  background$presence <- 0
  PA_data <- rbind(case_presence,background)
}

ext<-extract(s,PA_data[,1:2])
ext2<-cbind(PA_data,ext)

train_dt <- ext2 %>% filter(train == 1)
test_dt <- ext2 %>% filter(test_0 == 0)
train_dt2 <- train_dt[,c(5:ncol(train_dt))]
test_dt2 <- test_dt[,c(5:ncol(train_dt))]

# Create weight information, this step can be removed if unneccessary
weight_1 <-nrow(train_dt2)/(2*nrow(subset(train_dt2, presence == 1)))
weight_0 <-nrow(train_dt2)/(2*nrow(subset(train_dt2, presence == 0)))
weight_matrix <- ifelse(train_dt2$presence==1, weight_1*train_dt$BIAS_LAYER, #replace the BIAS_LAYER with actual column name
                        weight_0*train_dt$BIAS_LAYER) #replace the BIAS_LAYER

# Create Dmatrix to input covariate data into XGBoost model
# The dependent variable "presence" (indicate occurrence) and sampling bias variable "BIAS LAYER" were exxcluded from the covariate Dmatrix
train_Dmatrix <- train_dt2[,-1] %>%
  select(-c(presence, BIAS LAYER)) %>% #replace the BIAS_LAYER with actual column name
  as.matrix() %>% 
  xgb.DMatrix(weight=weight_matrix) 
test_Dmatrix <- test_dt2[,-1]
  select(-c(presence, BIAS LAYER)) %>% #replace the BIAS_LAYER with actual column name
  as.matrix() %>% 
  xgb.DMatrix()

targets <- as.factor(ifelse(train_dt2$presence==1, "y1", "y2"))  

# Run XGBoost model
set.seed(999)
model_xgb <- caret::train(
  objective = "binary:logistic",
  train_Dmatrix,targets,
  trControl = xgb_control_repcv,
  tuneGrid = xgb_grid,
  metric ="ROC",
  method = "xgbTree"
)

print(paste("XGB done ", dtset_i, sep=""))

# Turn the model prediction into raster
# The predicted raster is denoted as "new_res" here
result  <- raster::predict(model_xgb,
                          dmatrix_s,
                          na.rm =T, inf.rm = T,
                          type="prob")
res2 <- setValues(res,result$y1)
new_res <- mask(res2,overlap_mask)

print(paste("Prediction done ", dtset_i, sep=""))

#Set a file a directory for saving all raster output
writeRaster(new_res, filename = paste("XGB OUTPUT FILE LOCATION", 
                                      colnames(dataset)[dtset_i+2], 
                                      ".tif", sep = ""), overwrite=T)
            
# Input MaxEnt output raster
maxent_list <- list.files(path="MAXENT OUTPUT FILE LOCATION",
                    pattern = c("*.asc$"))
maxent_list
maxent_list2 <- Filter(function(x) !any(grepl("1km", x)), maxent_list)[1:30]
maxent_list2
setwd("MAXENT OUTPUT FILE LOCATION")
maxent_0 <- raster(maxent_list2[dtset_i])

# Create ensemble output by simple average of corresponding MaxEnt and XGBoost outputs                       
ensemble_0 <- (maxent_0 + new_res)/2
crs(ensemble_0) <- crs(new_res) 
                       
# Generate ensemble output raster and save it into a single folder
writeRaster(ensemble_0, filename = paste("ENSEMBLE OUTPUT FILE LOCATION",
                                      colnames(dataset)[dtset_i+2], 
                                      ".tif", sep = ""), overwrite=T)

print(paste("Ensemble done ", dtset_i, sep=""))

# Extract and assemble all actual/observed and predicted occurrence (dependent variable)
max_r_0 <-extract(maxent_0,PA_data[,1:2])
xgb_r_0 <-extract(new_res,PA_data[,1:2])
ens_r_0 <-extract(ensemble_0,PA_data[,1:2])
compare<-cbind(PA_data,max_r_0, xgb_r_0, ens_r_0)

# Save the assembled dataset for subsequent model performance comparison in PART 5
write.csv(compare, file = paste("ASSEMBLED MODEL OUTPUT LOCATION",
                                         colnames(dataset)[dtset_i+2], 
                                         ".csv", sep = ""), row.names = F)

print(paste("end ", dtset_i, sep=""))
}

#### PART 5: Model performance comparison
# List all assembled actual and predicted occurrence file 
compare_list <- list.files(path="ASSEMBLED MODEL OUTPUT LOCATION",
                    pattern = "*.csv$")
setwd("D:/Maxent Working directory 2.0/Comparison_table_output")
compare_list

#Calculate the evaluation mertrics (AUC, sensitivitiy, specificity, and F1) for both train and test dataset per model
for (file in compare_list[1:30]){
  compare <- read.csv(file, header=T, sep=",", stringsAsFactors=T)
  comp_tr <- compare %>% filter(train == 1)
  comp_te <- compare %>% filter(test_0 == 0)
  
  Aa <- roc(comp_tr$presence, comp_tr[,6])$auc
  Ab <- roc(comp_te$presence, comp_te[,6])$auc
  Ac <- roc(comp_tr$presence, comp_tr[,7])$auc
  Ad <- roc(comp_te$presence, comp_te[,7])$auc
  Ae <- roc(comp_tr$presence, comp_tr[,8])$auc
  Af <- roc(comp_te$presence, comp_te[,8])$auc
  
  CM_max_tr <- confusionMatrix(as.factor(round(comp_tr$max_r_0)),
                               as.factor(comp_tr$presence),
                               positive ="1")
  CM_max_te <- confusionMatrix(as.factor(round(comp_te$max_r_0)),
                               as.factor(comp_te$presence),
                               positive ="1")
  CM_xgb_tr <- confusionMatrix(as.factor(round(comp_tr$xgb_r_0)),
                               as.factor(comp_tr$presence),
                               positive ="1")
  CM_xgb_te <- confusionMatrix(as.factor(round(comp_te$xgb_r_0)),
                               as.factor(comp_te$presence),
                               positive ="1")
  CM_ens_tr <- confusionMatrix(as.factor(round(comp_tr$ens_r_0)),
                               as.factor(comp_tr$presence),
                               positive ="1")
  CM_ens_te <- confusionMatrix(as.factor(round(comp_te$ens_r_0)),
                               as.factor(comp_te$presence),
                               positive ="1")

  
  Ba <- CM_max_tr$byClass[1]
  Bb <- CM_max_te$byClass[1]
  Bc <- CM_xgb_tr$byClass[1]
  Bd <- CM_xgb_te$byClass[1]
  Be <- CM_ens_tr$byClass[1]
  Bf <- CM_ens_te$byClass[1]
  
  Ca <- CM_max_tr$byClass[2]
  Cb <- CM_max_te$byClass[2]
  Cc <- CM_xgb_tr$byClass[2]
  Cd <- CM_xgb_te$byClass[2]
  Ce <- CM_ens_tr$byClass[2]
  Cf <- CM_ens_te$byClass[2]
  
  Da <- CM_max_tr$byClass[7]
  Db <- CM_max_te$byClass[7]
  Dc <- CM_xgb_tr$byClass[7]
  Dd <- CM_xgb_te$byClass[7]
  De <- CM_ens_tr$byClass[7]
  Df <- CM_ens_te$byClass[7]

  # if the merged dataset does exist, append to it
  if (exists("Full_table")){
    Full_table <- rbind(Full_table, data.frame(Aa,Ab,Ac,Ad,Ae,Af,
                                               Ba,Bb,Bc,Bd,Be,Bf,
                                               Ca,Cb,Cc,Cd,Ce,Cf,
                                               Da,Db,Dc,Dd,De,Df, row.names = file))
  }
  # if the merged dataset doesn't exist, create it
  if (!exists("Full_table")){
    Full_table <-  data.frame(Aa,Ab,Ac,Ad,Ae,Af,
                              Ba,Bb,Bc,Bd,Be,Bf,
                              Ca,Cb,Cc,Cd,Ce,Cf,
                              Da,Db,Dc,Dd,De,Df, row.names = file)
  }
}

Full_table
Full_comparison <- as.data.frame(Full_table)
Full_comparison

# suffixes "max", "xgb", and "ens" indicate MaxEnt, XGBoost, and ensemble respectively
colnames(Full_comparison) <- c("auc_train_max", "auc_test_max",
                               "auc_train_xgb", "auc_test_xgb",
                               "auc_train_ens", "auc_test_ens",
                               "sens_train_max", "sens_test_max",
                               "sens_train_xgb", "sens_test_xgb",
                               "sens_train_ens", "sens_test_ens",
                               "spec_train_max", "spec_test_max",
                               "spec_train_xgb", "spec_test_xgb",
                               "spec_train_ens", "spec_test_ens",
                               "f1_train_max", "f1_test_max",
                               "f1_train_xgb", "f1_test_xgb",
                               "f1_train_ens", "f1_test_ens")

# Save the model performance comparison metrics
write.csv(Full_comparison, file = paste("ASSEMBLED MODEL OUTPUT LOCATION",
                                         "model_performance_comparison", 
                                         ".csv", sep = ""), row.names = F)                       

# Calculate mean and standard deviation (SD) or each metric column
column_mean <- apply(Full_comparison, 2, mean, na.rm = TRUE)
column_sd <- apply(Full_comparison, 2, mean, na.rm = TRUE)

summary_df <- rbind(column_mean, column_sd)
row.names(summary_df) <- c("mean", "SD")
colnames(summary_df) <- c("auc_train_max", "auc_test_max",
                               "auc_train_xgb", "auc_test_xgb",
                               "auc_train_ens", "auc_test_ens",
                               "sens_train_max", "sens_test_max",
                               "sens_train_xgb", "sens_test_xgb",
                               "sens_train_ens", "sens_test_ens",
                               "spec_train_max", "spec_test_max",
                               "spec_train_xgb", "spec_test_xgb",
                               "spec_train_ens", "spec_test_ens",
                               "f1_train_max", "f1_test_max",
                               "f1_train_xgb", "f1_test_xgb",
                               "f1_train_ens", "f1_test_ens")
summary_df
 
#### PART 6: Create mean output raster for each model 
#Average raster output
xgb_list <- list.files(path="XGB OUTPUT FILE LOCATION",
                    pattern = "*.tif$")
setwd("XGB OUTPUT FILE LOCATION")
xgb_rasters <- stack(paste0(xgb_list))
xgb_avg <- mean(xgb_rasters)
plot(xgb_avg)
writeRaster(xgb_avg, filename = paste("ASSEMBLED MODEL OUTPUT LOCATION",
                                      "XGBraster_mean",
                                       ".tif", sep = ""), overwrite=T)

ens_list <- list.files(path="ENSEMBLE OUTPUT FILE LOCATION",
                       pattern = "*.tif$")
setwd("ENSEMBLE OUTPUT FILE LOCATION")
ens_rasters <- stack(paste0(ens_list))
ens_avg <- mean(ens_rasters)
plot(ens_avg)
writeRaster(ens_avg, filename = paste("ASSEMBLED MODEL OUTPUT LOCATION",
                                      "Ensembleraster_mean",
                                      ".tif", sep = ""), overwrite=T)

#### PART 7: SHAP analysis for XGBoost model
#SHAP value for all dataset
for(dtset_i in 1:30){
print(paste("start ", dtset_i, sep=""))
{
  case_presence <- dataset[,c(1:2,dtset_i+2,dtset_i+2)]
  background <- dataset_bg[,c((dtset_i*2)-1, dtset_i*2)]
  colnames(case_presence)[1:4] <- c("X", "Y", "train","test_0")
  colnames(background)[1:2] <- c("X","Y")
  background$train <- 1
  background$test_0 <- 0
  case_presence$presence <- 1
  background$presence <- 0
  PA_data <- rbind(case_presence,background)
}

ext<-extract(s,PA_data[,1:2])
ext2<-cbind(PA_data,ext)

train_dt <- ext2 %>% filter(train == 1)
test_dt <- ext2 %>% filter(test_0 == 0)
train_dt2 <- train_dt[,c(5:ncol(train_dt))]
test_dt2 <- test_dt[,c(5:ncol(train_dt))]

# Create weight information, this step can be removed if unneccessary
weight_1 <-nrow(train_dt2)/(2*nrow(subset(train_dt2, presence == 1)))
weight_0 <-nrow(train_dt2)/(2*nrow(subset(train_dt2, presence == 0)))
weight_matrix <- ifelse(train_dt2$presence==1, weight_1*train_dt$BIAS_LAYER, #replace the BIAS_LAYER with actual column name
                        weight_0*train_dt$BIAS_LAYER) #replace the BIAS_LAYER

# Create Dmatrix to input covariate data into XGBoost model
# The dependent variable "presence" (indicate occurrence) and sampling bias variable "BIAS LAYER" were exxcluded from the covariate Dmatrix
train_Dmatrix <- train_dt2[,-1] %>%
  select(-c(presence, BIAS LAYER)) %>% #replace the BIAS_LAYER with actual column name
  as.matrix() %>% 
  xgb.DMatrix(weight=weight_matrix) 
test_Dmatrix <- test_dt2[,-1]
  select(-c(presence, BIAS LAYER)) %>% #replace the BIAS_LAYER with actual column name
  as.matrix() %>% 
  xgb.DMatrix()

targets <- as.factor(ifelse(train_dt2$presence==1, "y1", "y2"))  

# Run XGBoost model
set.seed(999)
model_xgb <- caret::train(
  objective = "binary:logistic",
  train_Dmatrix,targets,
  trControl = xgb_control_repcv,
  tuneGrid = xgb_grid,
  metric ="ROC",
  method = "xgbTree"
)

print(paste("XGB done ", dtset_i, sep=""))

# Start conducting SHAP analysis
print(paste("doing SHAP", dtset_i, sep="")) 
shap_values <- shap.values(xgb_model = model_xgb$finalModel, X_train = train_Dmatrix)
shap_values$mean_shap_score
  
if (exists("SHAP_table")){
    temp_SHAP <- data.frame(shap_values$mean_shap_score)
    SHAP_cov <- rownames(temp_SHAP)
    temp_SHAP2 <- data.frame(SHAP_cov, temp_SHAP)
    column_name <- colnames(dataset)[dtset_i+2]
    colnames(temp_SHAP2) <- c("covariate", column_name)
    SHAP_table <- left_join(SHAP_table, temp_SHAP2, by = c("covariate"="covariate"))
    rm(temp_SHAP, SHAP_cov, column_name, temp_SHAP2)
  }
  
if (!exists("SHAP_table")){
    temp_SHAP <- data.frame(shap_values$mean_shap_score)
    SHAP_cov <- rownames(temp_SHAP)
    SHAP_table <- data.frame(SHAP_cov, temp_SHAP)
    column_name <- colnames(dataset)[dtset_i+2]
    colnames(SHAP_table) <- c("covariate", column_name)
    rm(temp_SHAP, SHAP_cov, column_name)
  }
    
  print(paste("SHAP done ", dtset_i, sep=""))
}

# Save the merged table of generated SHAP values
write.csv(SHAP_table, filename = paste("ASSEMBLED MODEL OUTPUT LOCATION",
                                      "SHAP_table",
                                      ".tif", sep = ""), overwrite=T)
