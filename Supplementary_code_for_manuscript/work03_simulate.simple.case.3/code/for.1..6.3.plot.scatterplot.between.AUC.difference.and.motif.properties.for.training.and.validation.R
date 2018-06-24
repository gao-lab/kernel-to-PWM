library("plyr")
library("reshape2")
library("AUC")
library("doMC")
library("ggplot2")
library("plotROC")

all.PWMs.name.character.vector <- readLines("../data/for.1..2.all.PWMs.name.txt")


registerDoMC(cores=20)

PWM.name.and.PWM.type.and.auc.data.frame <- ldply(all.PWMs.name.character.vector, .fun=function(PWM.name.character){

    training.MLL.filename.character <- paste(sep="", "../data/for.1..5.MLL.for.motif.", PWM.name.character, ".and.dataset.training.fimo.txt")
    training.MLL.data.frame <- read.table(file=training.MLL.filename.character, header=TRUE, sep="\t", stringsAsFactors=FALSE)
    training.MLL.data.frame[, "PWM.type"] <- sub(pattern="(.*)_[0-9]+$", replacement="\\1", x=training.MLL.data.frame[, "PWM.name"])
    training.MLL.data.frame[, "y"] <- sub(pattern=".*__([01]+)$", replacement="\\1", x=training.MLL.data.frame[, "sequence.name"])

    validation.MLL.filename.character <- paste(sep="", "../data/for.1..5.MLL.for.motif.", PWM.name.character, ".and.dataset.validation.fimo.txt")
    validation.MLL.data.frame <- read.table(file=validation.MLL.filename.character, header=TRUE, sep="\t", stringsAsFactors=FALSE)
    validation.MLL.data.frame[, "PWM.type"] <- sub(pattern="(.*)_[0-9]+$", replacement="\\1", x=validation.MLL.data.frame[, "PWM.name"])
    validation.MLL.data.frame[, "y"] <- sub(pattern=".*__([01]+)$", replacement="\\1", x=validation.MLL.data.frame[, "sequence.name"])

    training.and.validation.MLL.data.frame <- rbind(training.MLL.data.frame,  validation.MLL.data.frame)
    

    PWM.type.and.auc.data.frame <- ddply(training.and.validation.MLL.data.frame, .variables="PWM.type", .fun=function(temp.data.frame){
        X.and.y.data.frame <- dcast(data=temp.data.frame, formula=sequence.name + y ~ PWM.name, value.var="score")
        X.matrix <- as.matrix(X.and.y.data.frame[, setdiff(colnames(X.and.y.data.frame), c("sequence.name", "y") ), drop=FALSE])
        y.factor.vector <- factor(X.and.y.data.frame[, "y"], levels=c("0", "1"))
        ## training.result.train <- NULL
        ## successfully.trained.flag.boolean <- FALSE
        ## set.seed(123)
        ## while(successfully.trained.flag.boolean == FALSE){
        ##     tryCatch({
        ##         training.result.train <- train(x=X.matrix, y=y.factor.vector, method="plr", trControl=trainControl(method="cv", number=5, allowParallel=FALSE))
        ##         successfully.trained.flag.boolean <- TRUE
        ##     }, error=function(e){cat(date(), " retrain PWM ", PWM.name.character, " with PWM type ", temp.data.frame[1, "PWM.type"], " due to the error: ", conditionMessage(e), "\n")})
        ## }
        ## auc.numeric <- auc(roc(predictions=predict(training.result.train, X.matrix, type="prob")[, 2], labels=y.factor.vector))
        ## return(c(auc=auc.numeric))
        auc.numeric <- auc(roc(predictions=X.matrix[, 1], labels=y.factor.vector))
        auc.reverse.numeric <- auc(roc(predictions=-1 * X.matrix[, 1], labels=y.factor.vector)) # this might be better if the expected logistic regression coefficient is negative
        return(c(auc=max(auc.numeric, auc.reverse.numeric)))
    })

    PWM.type.and.auc.data.frame[, "PWM.name"] <- PWM.name.character
    if ((which(all.PWMs.name.character.vector == PWM.name.character) %% 20) == 0){
        cat(date()," : Finished PWM ", PWM.name.character, "\n")
    }
    return(PWM.type.and.auc.data.frame)
}, .parallel=TRUE)

#PWM.name.and.PWM.type.and.auc.dcast.data.frame <- dcast(PWM.name.and.PWM.type.and.auc.data.frame, PWM.name ~ PWM.type, value.var="auc")


#hist(PWM.name.and.PWM.type.and.auc.dcast.data.frame[PWM.name.and.PWM.type.and.auc.dcast.data.frame[, "exp"] > 0.8, "exp"] -  PWM.name.and.PWM.type.and.auc.dcast.data.frame[PWM.name.and.PWM.type.and.auc.dcast.data.frame[, "exp"] > 0.8, "Deepbind"], plot=FALSE, breaks=c(-0.1, -0.05, -0.02, -0.01, -0.005, -0.002, -0.0001, 0, 0.1, 0.2, 0.3 ,0.4, 0.5, 0.6, 0.7))

PWM.name.and.kernel.auc.data.frame <- read.table(file="../data/for.1..3.3.PWM.name.and.training.and.validation.auc.txt", header=TRUE, sep="\t", stringsAsFactors=FALSE)




first.and.second.PWM.auc.difference.across.all.trainable.PWM.names.data.frame <- ddply(PWM.name.and.PWM.type.and.auc.data.frame, .variables="PWM.name", .fun=function(temp.data.frame){
#    first.and.second.PWM.auc.difference.data.frame <- adply(expand.grid(first.PWM.type=temp.data.frame[, "PWM.type"], second.PWM.type=temp.data.frame[, "PWM.type"], stringsAsFactors=FALSE), .margins=1, .fun=function(row.values.data.frame){
#        first.PWM.type.character <- row.values.data.frame[1, "first.PWM.type"]
#        second.PWM.type.character <- row.values.data.frame[1, "second.PWM.type"]
    first.and.second.PWM.auc.difference.data.frame <- ldply(combn(temp.data.frame[, "PWM.type"], m=2, simplify=FALSE),  .fun=function(temp.pair.character){
        first.PWM.type.character <- temp.pair.character[1]
        second.PWM.type.character <- temp.pair.character[2]
        return(data.frame(first.PWM.type=first.PWM.type.character, second.PWM.type=second.PWM.type.character, first.PWM.type.auc=temp.data.frame[temp.data.frame[, "PWM.type"]==first.PWM.type.character, "auc"], second.PWM.type.auc=temp.data.frame[temp.data.frame[, "PWM.type"]==second.PWM.type.character, "auc"], stringsAsFactors=FALSE ))
    })

    first.and.second.PWM.auc.difference.data.frame[, "PWM.name"] <- temp.data.frame[1, "PWM.name"]
    first.and.second.PWM.auc.difference.data.frame[, "original.kernel.auc"] <- PWM.name.and.kernel.auc.data.frame[PWM.name.and.kernel.auc.data.frame[, "PWM.name"] ==temp.data.frame[1, "PWM.name"], "training.and.validation.auc" ]
    return(first.and.second.PWM.auc.difference.data.frame)
}, .progress="text")
    
## write.table(x=first.and.second.PWM.auc.difference.across.all.trainable.PWM.names.data.frame, file="../data/for.1..6.3.first.and.second.PWM.auc.difference.across.all.trainable.PWM.names.txt", sep="\t", row.names=FALSE, col.names=TRUE, quote=FALSE)
## first.and.second.PWM.auc.difference.across.all.trainable.PWM.names.data.frame <- read.table(file="../data/for.1..6.3.first.and.second.PWM.auc.difference.across.all.trainable.PWM.names.txt", sep="\t", header=TRUE, stringsAsFactors=FALSE)

## png("../data/for.1..6.AUC.difference.plot.png", width=800, height=600)
## print( ggplot(first.and.second.PWM.auc.difference.across.all.trainable.PWM.names.data.frame, aes(x=paste(sep="", first.PWM.type, "\n-\n", second.PWM.type), y=first.PWM.type.auc - second.PWM.type.auc) ) + geom_boxplot() )
## dev.off()

png("../data/for.1..6.3.AUC.difference.plot.2.png", width=800, height=600)
print( ggplot(subset(first.and.second.PWM.auc.difference.across.all.trainable.PWM.names.data.frame, first.PWM.type=="Deepbind" & second.PWM.type == "exp"), aes(y= second.PWM.type.auc, x= first.PWM.type.auc, color=original.kernel.auc) ) + geom_point() + xlim(0.5, 1) + ylim(0.5, 1) + geom_abline(slope=1, intercept=0, color="red") + scale_color_gradientn(colors=rainbow(10)[1:8], limits=c(0.5, 1)) + theme(panel.background = element_rect(fill = 'black'), text=element_text(size=28)) + labs(x = "AUC of Deepbind's transformation", y = "AUC of our transformation",  color = "AUC of the original \n model (with kernels)"))
dev.off()



## to.plot.in.histogram.data.frame <- subset(first.and.second.PWM.auc.difference.across.all.trainable.PWM.names.data.frame, first.PWM.type=="Deepbind" & second.PWM.type == "exp")

## to.plot.in.histogram.with.difference.data.frame <- adply(.data=to.plot.in.histogram.data.frame, .margins=1, .fun=function(row.values.data.frame){
##     row.values.data.frame[, "difference"] <- row.values.data.frame[1, "second.PWM.type.auc"] - row.values.data.frame[1, "first.PWM.type.auc"]
##     row.values.data.frame[, "difference.clipped"] <- max(min(row.values.data.frame[, "difference"], 0.015), -0.015)
##     return(row.values.data.frame)
## })

## png("../data/for.1..6.3.AUC.difference.plot.2.in.histogram.png", width=800, height=600)
## print( ggplot(to.plot.in.histogram.with.difference.data.frame, aes(x=difference.clipped, fill=factor(floor(original.kernel.auc*20)/20)) ) + geom_histogram(binwidth=0.001) + geom_segment(x=0, y=0, xend=0, yend=Inf, color="black", size=2) + scale_fill_manual(values=rainbow(14)[1:11]) + theme(text=element_text(size=28)) + labs(x = "AUC of our transformation\n - AUC of Deepbind's transformation,\n restricted by [-0.015, 0.015]", fill = "AUC of the original \n model (with kernels)"))
## dev.off()

## png("../data/for.1..6.3.AUC.difference.plot.2.in.histogram.my.only.png", width=800, height=600)
## print( ggplot(to.plot.in.histogram.with.difference.data.frame, aes(x=difference.clipped, fill=factor(floor(original.kernel.auc*20)/20)) ) + geom_histogram(binwidth=0.001) + geom_segment(x=0, y=0, xend=0, yend=Inf, color="black", size=2) + scale_fill_manual(values=rainbow(14)[1:11]) + theme(text=element_text(size=28)) + labs(x = "AUC of my transformation\n - AUC of Deepbind's transformation,\n restricted by [-0.015, 0.015]", fill = "AUC of the original \n model (with kernels)"))
## dev.off()
