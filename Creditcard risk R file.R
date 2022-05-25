
#GROUP 19 DATA MINING PROJECT
#Saswati Mohanty, Vishal Vindyala, Urshilah Senthilnathan, Sujeeth Ganta, Sara Kamal

#CREDIT RISK ANALYSIS on creditcard transanction data with 492 frauds out of 284,807 transactions. 
#The dataset is highly unbalanced, 
#the positive class (frauds) account for 0.172% of all transactions.

#We have modelled the data for a series of models to achieve the lowest recall and 
#rightly account for the missclassification costs

#EXPLORATORY DATA ANALYSIS
#reading dataset
library(caret)
creditcard <- read.csv("~/R/creditcard datamining.csv", header=TRUE)
str(creditcard)
set.seed(12345)
creditcard$Class <- factor(creditcard$Class)
index <- createDataPartition(creditcard$Class, p = 0.75, list = F)
train <- creditcard[index, ]
test <- creditcard[-index, ]
#Class distribution bar plot
plot(creditcard$Class,xlab= "Class",ylab = "No. of Transaction")
#log(Amount) distribution
plot(log(creditcard$Amount))
# Create the histogram.
hist(creditcard$Amount,xlab = "Transaction",ylab = "Transaction Amount",col = "yellow",border = "blue", xlim = c(0,2000),ylim = c(0,60000),breaks = 50)

#MODEL 1: Classification Trees

#importing libraries
library(caret)
library(class)
library(e1071)

#reading dataset
creditcard <- read.csv("creditcard.csv", header=TRUE)
str(creditcard)
set.seed(12345)
creditcard$Class <- factor(creditcard$Class)
index <- createDataPartition(creditcard$Class, p = 0.75, list = F)
train <- creditcard[index, ]
test <- creditcard[-index, ]
#Using the Classification Tree model
library(tree)
tree.credit=tree(Class~.,train)
summary(tree.credit)
#summary shows the variables actually used in the model are:
#"V17" "V12" "V14" "V10" "V4" 
# To plot and all labels:
plot(tree.credit)
text(tree.credit,pretty=0)
tree.credit
tree.pred=predict(tree.credit,test,type="class")

# The confusion matrix (Prediction is rows, actual is columns)actual v/s predicted
(CMtree = table(test$Class,tree.pred))
(Acc = (CMtree[1,1]+CMtree[2,2])/sum(CMtree))
(recall = (CMtree[2,2]/(CMtree[2,1]+CMtree[2,2])))
#Introducing Missclassification costs

h=c(creditcard$Amount)
#Calculating Mean of Amount for each transaction
mean(h)
#Calculating MAx of Amount for each transaction
max(h)
# defining data in the vector for each cost matrix
x <- c(0,88.34962,88.34962,0)
y<-c(0,88.34962,25691.16,0)
z<-c(0,0,25691.16,0)
# defining row names and column names
rown <- c("0", "1")
coln <- c("0", "1")

# creating matrix
#Costmatrix with mean for Type1 and Type2 errors
Costmatrix <- matrix(x, nrow = 2, byrow = TRUE, 
                     dimnames = list(rown, coln))
#Costmatrix with mean for Type1 and Max for type2 errors
Costmatrix2<- matrix(y, nrow = 2, byrow = TRUE, 
                     dimnames = list(rown, coln))
#Costmatrix with zero for Type1 error and Max for Type2 errors
Costmatrix3<- matrix(z, nrow = 2, byrow = TRUE, 
                     dimnames = list(rown, coln))

# print matrix
Costmatrix
Costmatrix2
Costmatrix3
#Calculating cost matrix
MCtree1=Costmatrix*CMtree
MCtree1
MCtree2=Costmatrix2*CMtree
MCtree2
MCtree3=Costmatrix3*CMtree
MCtree3

#pruning tree
library(ISLR)
# We will set the seed because we will be doing cross-validation, which selects 
# partitions
#
# We will use cross-validation using the cv.tree function which will give us the 
# error rate for different tree sizes. The prune.misclass argument gives the number 
# of misclassification errors
#
cv.credit=cv.tree(tree.credit,FUN=prune.misclass)
names(cv.credit)

cv.credit
#parameters: 
#size of the tree, to be considered
#total no. of errors of each cross validation error - we want it further minimized i.e 69, of size 5
#k and method are not that relevant
# In the output, $size is the tree size and $dev is the number of errors. Pick the 
# The following plots help identify best size
plot(cv.credit$size,cv.credit$dev,type="b")
# Here we get 5 for best tree, but note that this number will depend on the seed set
# Now prune the tree to a best size of 5
prune.credit=prune.misclass(tree.credit,best=5)
plot(prune.credit)
text(prune.credit,pretty=0)
#5 leaf nodes and 4 splits
# Predict using the pruned tree
tree.pred=predict(prune.credit,test,type="class")
CMprune = table(test$Class,tree.pred)
CMprune
(Acc = (CMprune[1,1]+CMprune[2,2])/sum(CMprune))
(recall = (CMprune[2,2]/(CMprune[2,2]+CMprune[2,1])))
#Calculating cost matrix
MCprune=Costmatrix*CMprune
MCprune
MCprune2=Costmatrix2*CMprune
MCprune2
MCprune3=Costmatrix3*CMprune
MCprune3
#
# You can prune to any size you want by changing the best argument ...
prune.credit=prune.misclass(tree.credit,best=15)
plot(prune.credit)
text(prune.credit,pretty=0)
tree.pred=predict(prune.credit,test,type="class")
CMprune2 = table(tree.pred,test$Class)
CMprune2
(Acc = (CMprune2[1,1]+CMprune2[2,2])/sum(CMprune2))
(recall = (CMprune2[2,2]/(CMprune2[2,1]+CMprune2[2,2])))
#Calculating cost matrix
MCbestprune=Costmatrix*CMprune2
MCbestprune
MCbestprune2=Costmatrix2*CMprune2
MCbestprune2
MCbestprune3=Costmatrix3*CMprune2
MCbestprune3

#MODEL 2: KNN (K-Nearest Neighbors)

library(caret)
library(class)
library(e1071)
creditcard <- read.csv("~/R/creditcard datamining.csv", header=TRUE)

#Import dataset and pre-process it
str(creditcard)
set.seed(12345)
creditcard$Class <- factor(creditcard$Class)

index <- createDataPartition(creditcard$Class, p = 0.75, list = F)
train <- creditcard[index, ]
test <- creditcard[-index, ]
train_input <- as.matrix(train[,-31])
train_output <- train[,31]
test_input <- as.matrix(test[,-31])

#Model for k=5
# Prediction_Train
prediction <- knn(train_input, train_input,train_output, k=5)
# Prediction Test
prediction3 <- knn(train_input, test_input,train_output, k=5)
#Confusion Matrix
actual3 <- test$Class
(CM_Test <- table(actual3,prediction3))
CM_Test
#Accuracy and Recall
(Acc = (CM_Test[1,1]+CM_Test[2,2])/sum(CM_Test))
(recall = (CM_Test[2,2]/(CM_Test[2,1]+CM_Test[2,2])))


#Calculating cost matrix
MCKnn= Costmatrix*CM_Test
MCKnn
MCKnn2= Costmatrix2*CM_Test
MCKnn2
MCKnn3= Costmatrix3*CM_Test
MCKnn3

#MODEL 3: LOGISTIC REGRESSION

#importing libraries
library(caret)
library(class)
library(e1071)
#reading dataset
creditcarddata <- read.csv("~/R/creditcard datamining.csv", header=TRUE)
str(creditcarddata)
set.seed(12345)
creditcarddata$Class <- factor(creditcarddata$Class, levels = c("0","1"))
index <- createDataPartition(creditcarddata$Class, p = 0.75, list = F)
train <- creditcarddata[index, ]
test <- creditcarddata[-index, ]

# Logistic Model
fit1 <- glm(Class ~ ., data = train, family = "binomial")
summary(fit1)

# Generating Confusion Matrix at cutoff 0.5
actual.train <- train$Class
predicted.prob.train <- predict(fit1, data= train, type = "response")
cutoff <- 0.5

# Generate class predictions using cutoff value

predicted.train <- ifelse(predicted.prob.train > cutoff, "1","0")
predicted.train <- factor(predicted.train, levels = c("0","1"))

cm_train <- table(actual.train,predicted.train)
cm_train

# Validation on test data

actual.test <- test$Class
predicted.prob.test <- predict(fit1, newdata =test, type = "response")
cutoff <- 0.5

# Generate class predictions using cutoff value

predicted.test  <- ifelse(predicted.prob.test > cutoff, "1","0")
predicted.test <- factor(predicted.test, levels = c("0","1"))

# Confusion matrix (out of sample)

cm_test <- table(actual.test,predicted.test)
cm_test

(Acc = (cm_test[1,1]+cm_test[2,2])/sum(cm_test))
(recall = (cm_test[2,2]/(cm_test[2,1]+cm_test[2,2])))

#Calculating cost matrix
MCLogistic=Costmatrix*cm_test
MCLogistic
MCLogistic2=Costmatrix2*cm_test
MCLogistic2
MCLogistic3=Costmatrix3*cm_test
MCLogistic3


#MODEL 4: BOOSTING

df <- read.csv("~/R/creditcard datamining.csv")
df[,1] <- NULL
set.seed(12345)
inTrain <- createDataPartition(df$Class, p=0.75, list=FALSE)
dftrain <- data.frame(df[inTrain,])

dfvalidation <- data.frame(df[-inTrain,])
#Model of boosting
library(gbm)
boost.credit=gbm(Class ~.,data=dftrain,distribution="bernoulli",n.trees=100,interaction.depth=4)
summary(boost.credit)
par(mfrow=c(1,2))
yhat.boost=predict(boost.credit,newdata=dfvalidation,n.trees=100,type="response")
predicted <- ifelse(yhat.boost>=0.01,1,0)
yhat.test= dfvalidation$Class
#Confusion matrix
(c = table(predicted,yhat.test))
#Performance measures
(acc = (c[1,1]+c[2,2])/sum(c))
(sen = c[2,2]/(c[1,2]+c[2,2]))

#Calculating cost matrix
MCBoosting=Costmatrix*c
MCBoosting
MCBoosting2=Costmatrix2*c
MCBoosting2
MCBoosting3=Costmatrix3*c
MCBoosting3

## Naive Bayes

df <- read.csv("~/R/creditcard datamining.csv")
df[,1] <- NULL
df$Class <- as.factor(df$Class)
df$Amount <- as.factor(df$Amount)
library("caret")
set.seed(12345)

inTrain <- createDataPartition(df$Class, p=0.75, list=FALSE)
dftrain <- data.frame(df[inTrain,])

dfvalidation <- data.frame(df[-inTrain,])

library(e1071)
#Model for naive Bayes
model <- naiveBayes(Class~., data=dftrain)
model
prediction <- predict(model, newdata = dfvalidation[,-30],threshold = 0.4)
cm<-table(dfvalidation$Class,prediction,dnn=list('actual','predicted'))
model$apriori

predicted.probability <- predict(model, newdata = dfvalidation[,-30], type="raw")

PL <- as.numeric(dfvalidation$Class)-1
prob <- predicted.probability[,2]
df1 <- data.frame(prediction, PL, prob)

df1S <- df1[order(-prob),]
df1S$Gains <- cumsum(df1S$PL)
#Lift Chart
plot(df1S$Gains,type="n",main="Lift Chart",xlab="Number of Cases",ylab="Cumulative Success")
lines(df1S$Gains,col="blue")
abline(0,sum(df1S$PL)/nrow(df1S),lty = 2, col="red")
#Performance measures
(Acc = (cm[1,1]+cm[2,2])/sum(cm))
(recall = (cm[2,2]/(cm[2,1]+cm[2,2])))
## Misclassification costs
MCnaive=Costmatrix*cm
MCnaive
MCnaive2=Costmatrix2*cm
MCnaive2
MCnaive3=Costmatrix3*cm
MCnaive3

## Association rules

# Load the libraries
#library(arules)
#library(arulesViz)
#library(readxl)
#library(caret)
#library(class)
#library(e1071)
#library(tibble)
# Load the data set 
#creditcard <- read_xlsx("~/Desktop/Data Mining/DM group proj.xlsx")
#head(creditcard)
#creditcard <- as.data.frame(creditcard)
# Create an item frequency plot - part of arules package
#itemFrequencyPlot(creditcard,topN=15,type="relative")
# Get the rules
#rules <- apriori(creditcard, parameter = list(supp = 0.1, conf = 0.9))
#produces 3304 rules
# Show the top 5 rules
#inspect(rules[1:10])
#
#summary(rules)
#
#rules<-sort(rules, by="confidence", decreasing=TRUE)
#inspect(rules[1:20])
#
# For more concise rules - maxlen=3
#rules <- apriori(creditcard, parameter = list(supp = 0.1, conf = 0.9,maxlen=4))
#inspect(rules[1:5])
#
# Rules targeting (here rhs = whole milk)
#rules<-apriori(data=creditcard, parameter=list(supp=0.1,conf = 0.09), 
               #appearance = list(default="lhs",rhs="Class==1"),
               #control = list(verbose=F))
#rules<-sort(rules, decreasing=TRUE,by="confidence")
#inspect(rules[1:10])
#
#
# Visualization of rules
#library(arulesViz)
#plot(rules)
#plot(rules, measure=c("support", "lift"), shading="confidence")
#
#
#subrules <- rules[quality(rules)$confidence > 0.9]
#plot(subrules)
#

#BEST MODEL: Naive Bayes
library(ggplot2)
library(ROCR)
library(pROC)
plot(roc(dfvalidation$Class, predicted.probability, direction="<"),
               col="yellow", lwd=3, main="The")
data_balanced_under <- ovun.sample(class ~ ., data = creditcarddata, method = "under", N = 40, seed = 1)$data
table(data_balanced_under$class)
########################################
#Under-sampling
summary(creditcarddata$Class)
predictor_variables <- creditcarddata[,-31] # Select everything except response
response_variable <- creditcarddata$Class   # Only select response variable
# swaps around the factor encoding for legitimate and fraud
levels(response_variable) <- c('0', '1') 

# Run undersampled function

library(unbalanced)
undersampled_data <- ubBalance(predictor_variables, 
                               response_variable, 
                               type='ubUnder',         # Option for undersampling
                               verbose = TRUE)
undersampled_combined <- cbind(undersampled_data$X,    # combine output
                               undersampled_data$Y)
undersampled_combined

names(undersampled_combined)[names(undersampled_combined) == "undersampled_data$Y"] <- "Class" # change name to class
levels(undersampled_combined$Class) <- c('Legitimate', 'Fraud')
# plot number of cases in undersampled dataset
dev.off()
ggplot(data = undersampled_combined, aes(fill = Class), color = "#AA6666")+
  geom_bar(aes(x = Class), color = "#AA6666")+
  ggtitle("Class Distribution after undersampling", 
          subtitle="Total samples: 984")+
  xlab("")+
  ylab("Samples")+
  scale_y_continuous(expand = c(0,0))+
  scale_x_discrete(expand = c(0,0))+
  theme(legend.position = "none", 
        legend.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank())
undersampled_combined[,1] <- NULL
undersampled_combined$Class <- as.factor(undersampled_combined$Class)
undersampled_combined$Amount <- as.factor(undersampled_combined$Amount)
inTrain <- createDataPartition(undersampled_combined$Class, p=0.75, list=FALSE)
dftrain <- data.frame(undersampled_combined[inTrain,])
dfvalidation <- data.frame(undersampled_combined[-inTrain,])

#Best model with under-sampling
library(e1071)

model <- naiveBayes(Class~., data=dftrain)
model
prediction <- predict(model, newdata = dfvalidation[,-30],threshold = 0.4)
cm_best_model<-table(dfvalidation$Class,prediction,dnn=list('actual','predicted'))
model$apriori

predicted.probability <- predict(model, newdata = dfvalidation[,-30], type="raw")

PL <- as.numeric(dfvalidation$Class)-1
prob <- predicted.probability[,2]
df1 <- data.frame(prediction, PL, prob)

df1S <- df1[order(-prob),]
df1S$Gains <- cumsum(df1S$PL)
plot(df1S$Gains,type="n",main="Lift Chart",xlab="Number of Cases",ylab="Cumulative Success")
lines(df1S$Gains,col="blue")
abline(0,sum(df1S$PL)/nrow(df1S),lty = 2, col="red")

## Performance measures and Misclassification costs

(Acc = (cm[1,1]+cm[2,2])/sum(cm))
(recall = (cm[2,2]/(cm[2,1]+cm[2,2])))

MCbest_model=Costmatrix*cm
MCbest_model
MCbest_model2=Costmatrix2*cm
MCbest_model2
MCbest_model3=Costmatrix3*cm
MCbest_model3

