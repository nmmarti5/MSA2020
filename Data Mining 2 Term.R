options(digits = 2)
#Added a binary for term or not and I deleted y and next product.
TermData <- read.csv(file="C:/Users/17708/Documents/R/TermData.csv", header=TRUE, sep=",")
set.seed(12345)
perm2=sample(1:41188)
Bank_random0rder2=TermData[perm2,]
train2 = Bank_random0rder2[1:floor(0.75*41188),]
test2 = Bank_random0rder2[(floor(0.75*41188)+1):41188,]
library("rpart")
tree2 = rpart(Term~ . -Term, data=train2, method='class',
             parms = list(split='entrophy'))
.pardefault = par()
par(mai=c(.2,.2,.2,.2))
plot(tree2, uniform=T)
text(tree2)
#text(tree2, use.n=T)
par(.pardefault)
tree2$variable.importance
library('lattice')
barchart(tree2$variable.importance[order(tree2$variable.importance)],
         xlab = 'Importance', horiz=T, xlim=c(0,2000),ylab='Variable',
         main = 'Variable Importance',cex.names=0.8, las=2, col = 'orange')
tscores2 = predict(tree2,type='class')
scores2 = predict(tree2, test2, type='class')
cat('Training Misclassification Rate:',
    + sum(tscores2!=train2$Term)/nrow(train2))
cat('Validation Misclassification Rate:',
    sum(scores2!=test2$Term)/nrow(test2))
library("rattle") # Fancy tree2 plot
library("rpart.plot") # Enhanced tree2 plots
library("RColorBrewer") # Color selection for fancy tree2 plot
library("party") # Alternative decision tree2 algorithm
library("partykit") # Convert rpart object to Binarytree2
prp(tree2, type =0, extra=8, leaf.round=1, border.col=1,
    box.col=brewer.pal(10,"Set3")[tree2$frame$yval], )



