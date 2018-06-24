## Summary of steps
## 9. compare the MAPE and MSE between two transformations, and plot the result

library("ggplot2")
library("plyr")

motif.id.character.vector <- c("D00600.001", "D00600.004", "D00350.002", "D00350.005", "D00558.002", "D00328.003", "D00694.003", "D00588.002")

comparison.performance.result.for.all.motifs.data.frame <- ldply(.data=motif.id.character.vector, .fun=function(temp.motif.id.character){
    comparison.result.result.filename.character <- paste(sep="", "../data/for.1..2.4.testing.prediction.and.real.for.motif.", temp.motif.id.character, ".txt")
    if ( file.exists(comparison.result.result.filename.character) == FALSE ){
        return(NULL)
    }
    comparison.result.result.data.frame <- read.table(file=comparison.result.result.filename.character, sep="\t", header=TRUE)
    sample.count.integer <- nrow(comparison.result.result.data.frame)
    MSE.exp.PFM.numeric <- sum((comparison.result.result.data.frame[, "exp.PFM"] - comparison.result.result.data.frame[, "real"])^2) / sample.count.integer
    MSE.Deepbind.PFM.numeric <- sum((comparison.result.result.data.frame[, "Deepbind.PFM"] - comparison.result.result.data.frame[, "real"])^2) / sample.count.integer
    MAPE.exp.PFM.numeric <- sum(abs(
    (comparison.result.result.data.frame[, "exp.PFM"] - comparison.result.result.data.frame[, "real"] ) / comparison.result.result.data.frame[, "real"]
    )) / sample.count.integer
    MAPE.Deepbind.PFM.numeric <- sum(abs(
    (comparison.result.result.data.frame[, "Deepbind.PFM"] - comparison.result.result.data.frame[, "real"] ) / comparison.result.result.data.frame[, "real"]
    )) / sample.count.integer

    return(data.frame(
        motif.id=temp.motif.id.character,
        measurement=c("MSE", "MAPE"),
        exp.PFM.value=c(MSE.exp.PFM.numeric, MAPE.exp.PFM.numeric),
        Deepbind.PFM.value=c(MSE.Deepbind.PFM.numeric, MAPE.Deepbind.PFM.numeric),
        stringsAsFactors=FALSE
    ))
})


maximal.exp.number.to.plot.numeric <- max(comparison.performance.result.for.all.motifs.data.frame[, "exp.PFM.value"])
maximal.Deepbind.number.to.plot.numeric <- max(comparison.performance.result.for.all.motifs.data.frame[, "Deepbind.PFM.value"])


motif.id.to.name.named.character.vector <- c(
    "D00600.001"="PRDM1 (SELEX)",
    "D00600.004"="PRDM1 (ChIP)",
    "D00350.002"="EBF1 (SELEX)",
    "D00350.005"="EBF1 (ChIP)",
    "D00558.002"="NR4A2 (SELEX)",
    "D00328.003"="CTCF (SELEX)",
    "D00694.003"="Tp53 (SELEX)",
    "D00588.002"="Pou2f2 (SELEX)"
)

png(file="../data/for.1..2.6.MAPE.and.MSE.comparison.result.png", width=800, height=600)
ggplot(comparison.performance.result.for.all.motifs.data.frame, aes(x=exp.PFM.value, y=Deepbind.PFM.value, color=motif.id.to.name.named.character.vector[motif.id]))  + geom_point(size=5, shape=13) + geom_abline(slope=1, intercept=0, color="red") + facet_wrap(facets=~measurement, scales="free") + labs(x="Loss of retrained model with log-likelihood of\nour PWM", y="Loss of retrained model with log-likelihood of\nDeepbind's PWM", color="Motif (experiment)") + scale_color_manual(values=c("black", "red", "orange", "brown", "green", "blue", "purple")) + theme(text=element_text(size=18))
dev.off()
