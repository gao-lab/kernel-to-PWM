library("ggplot2")
library("plyr")

input.command.line.argument.data.frame <- read.table(file="../data/for.2..1.input.command.line.argument.txt", sep="\t", header=TRUE, stringsAsFactors=FALSE)

motif.id.character.vector <- input.command.line.argument.data.frame[, "ID"]

comparison.performance.result.for.all.motifs.data.frame <- ldply(.data=motif.id.character.vector, .fun=function(temp.motif.id.character){
    stop.flag.filename.character <- paste(sep="", "../data/for.2..2.1.stop.flag.for.motif.", temp.motif.id.character, ".txt")
    comparison.result.result.filename.character <- paste(sep="", "../data/for.2..2.4.testing.prediction.and.real.for.motif.", temp.motif.id.character, ".txt")
    pseudotraining.result.filename.character <- paste(sep="", "../data/for.2..2.5.testing.prediction.and.real.for.motif.", temp.motif.id.character, ".with.pseudotraining.txt")
    if (file.exists(stop.flag.filename.character) == TRUE){
        return(NULL)
    }
    else if ( file.exists(pseudotraining.result.filename.character) == FALSE){
        return(NULL)
    }
    else if ( file.exists(comparison.result.result.filename.character) == FALSE ){
        return(NULL)
    }
    comparison.result.result.data.frame <- read.table(file=comparison.result.result.filename.character, sep="\t", header=TRUE)
    pseudotraining.result.data.frame <- read.table(file=pseudotraining.result.filename.character, sep="\t", header=TRUE)
    sample.count.integer <- nrow(comparison.result.result.data.frame)
    MSE.exp.PFM.numeric <- sum((comparison.result.result.data.frame[, "exp.PFM"] - comparison.result.result.data.frame[, "real"])^2) / sample.count.integer
    MSE.Deepbind.PFM.numeric <- sum((comparison.result.result.data.frame[, "Deepbind.PFM"] - comparison.result.result.data.frame[, "real"])^2) / sample.count.integer
    MSE.exp.PFM.pseudotraining.numeric <- sum((pseudotraining.result.data.frame[, "exp.PFM.pseudotraining"] - pseudotraining.result.data.frame[, "real"])^2) / nrow(pseudotraining.result.data.frame)
    MAPE.exp.PFM.numeric <- sum(abs(
    (comparison.result.result.data.frame[, "exp.PFM"] - comparison.result.result.data.frame[, "real"] ) / comparison.result.result.data.frame[, "real"]
    )) / sample.count.integer
    MAPE.Deepbind.PFM.numeric <- sum(abs(
    (comparison.result.result.data.frame[, "Deepbind.PFM"] - comparison.result.result.data.frame[, "real"] ) / comparison.result.result.data.frame[, "real"]
    )) / sample.count.integer
    MAPE.exp.PFM.pseudotraining.numeric <- sum(abs(
    (pseudotraining.result.data.frame[, "exp.PFM.pseudotraining"] - pseudotraining.result.data.frame[, "real"] ) / pseudotraining.result.data.frame[, "real"]
    )) / nrow(pseudotraining.result.data.frame)

    return(data.frame(
        motif.id=temp.motif.id.character,
        measurement=c("MSE", "MAPE", "MSE", "MAPE"),
        PFM.type=c("ours", "ours", "Deepbind", "Deepbind"),
        value=c(MSE.exp.PFM.pseudotraining.numeric, MAPE.exp.PFM.pseudotraining.numeric, MSE.Deepbind.PFM.numeric, MAPE.Deepbind.PFM.numeric),
        stringsAsFactors=FALSE
    ))
}, .progress="text")




png(file="../data/for.2..3.MAPE.and.MSE.comparison.result.for.all.motifs.png", width=800, height=600)
ggplot(comparison.performance.result.for.all.motifs.data.frame, aes(x=PFM.type, y=log10(value)))  + geom_boxplot() + facet_wrap(facets=~measurement, scales="free") + labs(x="Type of PWM", y="log10 of loss of retrained model", color="Motif (experiment)") + theme(text=element_text(size=18))
dev.off()
