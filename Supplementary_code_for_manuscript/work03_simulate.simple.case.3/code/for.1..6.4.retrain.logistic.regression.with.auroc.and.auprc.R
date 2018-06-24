library("caret")
library("plyr")
library("reshape2")
library("AUC")
library("doMC")
library("stepPlr")
library("PerfMeas")

all.PWMs.name.character.vector <- readLines("../data/for.1..2.all.PWMs.name.txt")


registerDoMC(cores=30)

PWM.name.and.PWM.type.and.auc.data.frame <- ldply(all.PWMs.name.character.vector, .fun=function(PWM.name.character){

    training.MLL.filename.character <- paste(sep="", "../data/for.1..5.MLL.for.motif.", PWM.name.character, ".and.dataset.training.fimo.txt")
    validation.MLL.filename.character <- paste(sep="", "../data/for.1..5.MLL.for.motif.", PWM.name.character, ".and.dataset.validation.fimo.txt")

    training.MLL.data.frame <- read.table(file=training.MLL.filename.character, header=TRUE, sep="\t", stringsAsFactors=FALSE)
    validation.MLL.data.frame <- read.table(file=validation.MLL.filename.character, header=TRUE, sep="\t", stringsAsFactors=FALSE)

    training.and.validation.MLL.data.frame <- rbind(training.MLL.data.frame, validation.MLL.data.frame)
    training.and.validation.MLL.data.frame[, "y"] <- sub(pattern=".*__([01]+)$", replacement="\\1", x=training.and.validation.MLL.data.frame[, "sequence.name"])
    training.and.validation.MLL.data.frame[, "PWM.type"] <- sub(pattern="(.*)_[0-9]+$", replacement="\\1", x=training.and.validation.MLL.data.frame[, "PWM.name"])

    PWM.type.and.auroc.and.auprc.data.frame <- ddply(training.and.validation.MLL.data.frame, .variables="PWM.type", .fun=function(temp.data.frame){
        X.and.y.data.frame <- dcast(data=temp.data.frame, formula=sequence.name + y ~ PWM.name, value.var="score")
        X.matrix <- as.matrix(X.and.y.data.frame[, setdiff(colnames(X.and.y.data.frame), c("sequence.name", "y") ), drop=FALSE])
        y.factor.vector <- factor(X.and.y.data.frame[, "y"], levels=c("0", "1"))
        training.result.train <- NULL
        successfully.trained.flag.boolean <- FALSE
        set.seed(123)
        while(successfully.trained.flag.boolean == FALSE){
            tryCatch({
                training.result.train <- train(x=X.matrix, y=y.factor.vector, method="plr", trControl=trainControl(method="cv", number=5, allowParallel=FALSE))
                successfully.trained.flag.boolean <- TRUE
            }, error=function(e){cat(date(), " retrain PWM ", PWM.name.character, " with PWM type ", temp.data.frame[1, "PWM.type"], " due to the error: ", conditionMessage(e), "\n")})
        }
        temp.prediction.factor.vector <- predict(training.result.train, X.matrix)
        auroc.numeric <- auc(roc(predictions=temp.prediction.factor.vector, labels=y.factor.vector))
        temp.prediction.probability.numeric.vector <- predict(training.result.train, X.matrix, type="prob")[, "1"]
        auprc.numeric <- AUPRC(list(precision.at.all.recall.levels(scores=temp.prediction.probability.numeric.vector, labels=as.integer(as.character(y.factor.vector)))))
        return(c(auroc=auroc.numeric, auprc=auprc.numeric))
    })

    PWM.type.and.auroc.and.auprc.data.frame[, "PWM.name"] <- PWM.name.character
    if ((which(all.PWMs.name.character.vector == PWM.name.character) %% 20) == 0){
        cat(date()," : Finished PWM ", PWM.name.character, "\n")
    }
    return(PWM.type.and.auroc.and.auprc.data.frame)
}, .parallel=TRUE)


colnames(PWM.name.and.PWM.type.and.auc.data.frame) <- c("PWM.type", "auroc", "auprc", "PWM.name")

PWM.name.and.kernel.auc.data.frame <- read.table(file="../data/for.1..3.3.PWM.name.and.training.and.validation.auc.txt", header=TRUE, sep="\t", stringsAsFactors=FALSE)


first.and.second.PWM.auc.across.all.PWM.names.data.frame <- ddply(PWM.name.and.PWM.type.and.auc.data.frame, .variables="PWM.name", .fun=function(temp.data.frame){
    first.and.second.PWM.auc.data.frame <- ldply(combn(temp.data.frame[, "PWM.type"], 2, simplify=FALSE), .fun=function(first.and.second.PWM.type.character.vector){
        first.PWM.type.character <- first.and.second.PWM.type.character.vector[1]
        second.PWM.type.character <- first.and.second.PWM.type.character.vector[2]
        return(data.frame(first.PWM.type=first.PWM.type.character,
                          second.PWM.type=second.PWM.type.character,
                          first.PWM.value=c(temp.data.frame[temp.data.frame[, "PWM.type"]==first.PWM.type.character, "auroc"], temp.data.frame[temp.data.frame[, "PWM.type"]==first.PWM.type.character, "auprc"]),
                          second.PWM.value=c(temp.data.frame[temp.data.frame[, "PWM.type"]==second.PWM.type.character, "auroc"], temp.data.frame[temp.data.frame[, "PWM.type"]==second.PWM.type.character, "auprc"]),
                          AUC.type=c("AUROC", "AUPRC"),
                          stringsAsFactors=FALSE ))
    })

    first.and.second.PWM.auc.data.frame[, "PWM.name"] <- temp.data.frame[1, "PWM.name"]
    first.and.second.PWM.auc.data.frame[, "original.kernel.auc"] <- subset(PWM.name.and.kernel.auc.data.frame, PWM.name == temp.data.frame[1, "PWM.name"])[1, "training.and.validation.auc"]
    
    return(first.and.second.PWM.auc.data.frame)
})
    


write.table(first.and.second.PWM.auc.across.all.PWM.names.data.frame, "../data/for.1..6.4.first.and.second.PWM.auc.across.all.PWM.names.txt", sep="\t", row.names=FALSE, col.names=TRUE, quote=FALSE)

## first.and.second.PWM.auc.across.all.PWM.names.data.frame <- read.table("../data/for.1..6.4.first.and.second.PWM.auc.across.all.PWM.names.txt", header=TRUE)



png("../data/for.1..6.4.AUROC.and.AUPRC.difference.plot.2.png", width=1000, height=600)
print( ggplot(subset(first.and.second.PWM.auc.across.all.PWM.names.data.frame, first.PWM.type=="Deepbind" & second.PWM.type == "exp"), aes(y= second.PWM.value, x= first.PWM.value) ) + geom_point() + xlim(0.5, 1) + ylim(0.5, 1) + geom_abline(slope=1, intercept=0, color="red") + scale_color_gradientn(colors=rainbow(10)[1:8], limits=c(0.5, 1)) + theme(text=element_text(size=28)) + facet_wrap(~AUC.type) + labs(x = "AUC of Deepbind's transformation", y = "AUC of the exact transformation",  color = "AUC of the original \n model (with kernels)"))
dev.off()

## histogram version

png("../data/for.1..6.4.AUROC.and.AUPRC.difference.plot.2.histogram.png", width=1000, height=600)
print( ggplot(subset(first.and.second.PWM.auc.across.all.PWM.names.data.frame, first.PWM.type=="Deepbind" & second.PWM.type == "exp"), aes(x= second.PWM.value - first.PWM.value ) ) + geom_vline(xintercept=0, color="red") + geom_histogram() + theme(text=element_text(size=28)) + facet_wrap(~AUC.type) + labs(x = "AUC of the exact transformation minus\nAUC of Deepbind's transformation", y="count"))
dev.off()


for.AUPRC.statistical.test.data.frame <- subset(first.and.second.PWM.auc.across.all.PWM.names.data.frame, first.PWM.type=="Deepbind" & second.PWM.type == "exp" & AUC.type == "AUPRC")
for.AUROC.statistical.test.data.frame <- subset(first.and.second.PWM.auc.across.all.PWM.names.data.frame, first.PWM.type=="Deepbind" & second.PWM.type == "exp" & AUC.type == "AUROC")

wilcox.test(for.AUPRC.statistical.test.data.frame[, "second.PWM.value"], for.AUPRC.statistical.test.data.frame[, "first.PWM.value"], alternative="greater", paired = TRUE)
wilcox.test(for.AUROC.statistical.test.data.frame[, "second.PWM.value"], for.AUROC.statistical.test.data.frame[, "first.PWM.value"], alternative="greater", paired = TRUE)
