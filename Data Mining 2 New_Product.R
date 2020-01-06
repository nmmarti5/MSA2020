#Completed for Dr. Shaina Race's Data Mining Course with assistance from code she provided 
options(digits = 2)
#Removed Y var.
bank <- read.csv(file="C:/Users/17708/Documents/R/bankdata_new2.csv", header=TRUE, sep=",")
set.seed(12345)
perm=sample(1:41188)
Bank_random0rder=bank[perm,]
train = Bank_random0rder[1:floor(0.75*41188),]
test = Bank_random0rder[(floor(0.75*41188)+1):41188,]
library("rpart")
tree = rpart(next.product~ . -next.product, data=train, method='class',
             parms = list(split='entrophy'),
             control = rpart.control(minbucket = 500, maxdepth = 5))
.pardefault = par()
par(mai=c(.2,.2,.2,.2))
plot(tree, uniform=T)
text(tree)
#text(tree, use.n=T)
par(.pardefault)
tree$variable.importance
library('lattice')
barchart(tree$variable.importance[order(tree$variable.importance)],
         xlab = 'Importance', horiz=T, xlim=c(0,2000),ylab='Variable',
         main = 'Variable Importance',cex.names=0.8, las=2, col = 'orange')
tscores = predict(tree,type='class')
scores = predict(tree, test, type='class')
cat('Training Misclassification Rate:',
    + sum(tscores!=train$next.product)/nrow(train))
cat('Validation Misclassification Rate:',
    sum(scores!=test$next.product)/nrow(test))
library("rattle") # Fancy tree plot
library("rpart.plot") # Enhanced tree plots
library("RColorBrewer") # Color selection for fancy tree plot
library("party") # Alternative decision tree algorithm
library("partykit") # Convert rpart object to BinaryTree
prp(tree, type =0, extra=8, leaf.round=1, border.col=1,
    box.col=brewer.pal(10,"Set3")[tree$frame$yval], )



