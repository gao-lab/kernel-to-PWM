library("caret")
library("plyr")
library("reshape2")
library("AUC")
library("doMC")
library("stepPlr")

all.PWMs.name.character.vector <- readLines("../data/for.1..2.all.PWMs.name.txt")


registerDoMC(cores=30)

PWM.name.and.PWM.type.and.auc.data.frame <- ldply(all.PWMs.name.character.vector, .fun=function(PWM.name.character){

    training.MLL.filename.character <- paste(sep="", "../data/for.1..5.MLL.for.motif.", PWM.name.character, ".and.dataset.training.fimo.txt")
    validation.MLL.filename.character <- paste(sep="", "../data/for.1..5.MLL.for.motif.", PWM.name.character, ".and.dataset.validation.fimo.txt")

    training.MLL.data.frame <- read.table(file=training.MLL.filename.character, header=TRUE, sep="\t", stringsAsFactors=FALSE)
    validation.MLL.data.frame <- read.table(file=validation.MLL.filename.character, header=TRUE, sep="\t", stringsAsFactors=FALSE)

    training.and.validation.MLL.data.frame <- rbind(training.MLL.data.frame, validation.MLL.data.frame)
    training.and.validation.MLL.data.frame[, "y"] <- sub(pattern=".*__([01]+)$", replacement="\\1", x=training.and.validation.MLL.data.frame[, "sequence.name"])
    training.and.validation.MLL.data.frame[, "PWM.type"] <- sub(pattern="(.*)_[0-9]+$", replacement="\\1", x=training.and.validation.MLL.data.frame[, "pattern.name"])

    PWM.type.and.auc.data.frame <- ddply(training.and.validation.MLL.data.frame, .variables="PWM.type", .fun=function(temp.data.frame){
        X.and.y.data.frame <- dcast(data=temp.data.frame, formula=sequence.name + y ~ pattern.name, value.var="score")
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
        auc.numeric <- auc(roc(predictions=predict(training.result.train, X.matrix), labels=y.factor.vector))
        return(c(auc=auc.numeric))
    })

    PWM.type.and.auc.data.frame[, "PWM.name"] <- PWM.name.character
    if ((which(all.PWMs.name.character.vector == PWM.name.character) %% 20) == 0){
        cat(date()," : Finished PWM ", PWM.name.character, "\n")
    }
    return(PWM.type.and.auc.data.frame)
}, .parallel=TRUE)


first.and.second.PWM.auc.difference.across.all.PWM.names.data.frame <- ddply(PWM.name.and.PWM.type.and.auc.data.frame, .variables="PWM.name", .fun=function(temp.data.frame){
    first.and.second.PWM.auc.difference.data.frame <- ldply(combn(temp.data.frame[, "PWM.type"], 2, simplify=FALSE), .fun=function(first.and.second.PWM.type.character.vector){
        first.PWM.type.character <- first.and.second.PWM.type.character.vector[1]
        second.PWM.type.character <- first.and.second.PWM.type.character.vector[2]
        return(data.frame(first.PWM.type=first.PWM.type.character, second.PWM.type=second.PWM.type.character, auc.difference=temp.data.frame[temp.data.frame[, "PWM.type"]==first.PWM.type.character, "auc"] - temp.data.frame[temp.data.frame[, "PWM.type"]==second.PWM.type.character, "auc"], stringsAsFactors=FALSE ))
    })

    first.and.second.PWM.auc.difference.data.frame[, "PWM.name"] <- temp.data.frame[1, "PWM.name"]
    return(first.and.second.PWM.auc.difference.data.frame)
})
    
