#Loading libraries & data
bcdata <- read.csv("BreastCancerData.csv")
head(bcdata)

bcdata <- bcdata[, -1]

summary(bcdata)
bcdata$diagnosis1 <- factor(bcdata$diagnosis1)
sapply(bcdata, class)

#PCA - Did not use as I decided to make a RF
library(factoextra)
bcdata_pca <- transform(bcdata) 
all_pca <- prcomp(bcdata_pca[,-1], cor=TRUE, scale = TRUE)
summary(all_pca)

fviz_eig(all_pca, addlabels=TRUE, ylim=c(0,50), geom = c("bar", "line"), barfill = "orange", barcolor="gray",linecolor = "blue", ncp=10)+
  labs(title = "Breast Cancer All Variances - PCA",
       x = "Principal Components", y = "% of variances")

all_var <- get_pca_var(all_pca)
all_var


#Train & Test Split
set.seed(12345)
train_id <- sample(seq_len(nrow(bcdata)), size = floor(0.80*nrow(bcdata)))

train <- bcdata[train_id, ]
test <- bcdata[-train_id, ]

table(train$diagnosis1)
table(test$diagnosis1)

#Random Forest - Train
#install.packages("randomForest")
library(randomForest)

rf <- randomForest(diagnosis1 ~ ., data = train, ntree = 75, proximity = T, importance = T)
rf
plot(rf)
importance(rf)
print(rf)

# Variable Importance
varImpPlot(rf,  
           sort = T,
           n.var=10,
           main="Top 10 - Variable Importance")

#Random Forest - Test
rfPred = predict(rf, newdata = test)
table(rfPred, test$diagnosis1)
plot(margin(rf, testa$diagnosis1))
CM = table(rfPred, test$diagnosis1)
accuracy = (sum(diag(CM)))/sum(CM) #93%

