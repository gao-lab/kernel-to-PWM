library("ggplot2")
library("reshape2")
library("plyr")

AUC.summary.data.frame <- read.table("../data/for.2..8.2.AUROC.and.AUPRC.summary.for.Rfam.txt", header=TRUE, sep="\t", stringsAsFactors=FALSE)

AUC.summary.paired.data.frame <- ddply(AUC.summary.data.frame, .variables="CM.accession", .fun=function(temp.data.frame){
    cross.comparison.data.frame <- adply(expand.grid(first.PWM.type=temp.data.frame[, "PWM.type"], second.PWM.type=temp.data.frame[, "PWM.type"], stringsAsFactors=FALSE), .margins=1, .fun=function(row.values.data.frame){
        first.PWM.type.character <- row.values.data.frame[1, "first.PWM.type"]
        second.PWM.type.character <- row.values.data.frame[1, "second.PWM.type"]
        first.PWM.type.AUROC.numeric <- temp.data.frame[temp.data.frame[, "PWM.type"] == first.PWM.type.character, "AUROC"]
        second.PWM.type.AUROC.numeric <- temp.data.frame[temp.data.frame[, "PWM.type"] == second.PWM.type.character, "AUROC"]
        first.PWM.type.AUPRC.numeric <- temp.data.frame[temp.data.frame[, "PWM.type"] == first.PWM.type.character, "AUPRC"]
        second.PWM.type.AUPRC.numeric <- temp.data.frame[temp.data.frame[, "PWM.type"] == second.PWM.type.character, "AUPRC"]
        return(data.frame(first.PWM.type=first.PWM.type.character, second.PWM.type=second.PWM.type.character, first.PWM.value=c(first.PWM.type.AUROC.numeric, first.PWM.type.AUPRC.numeric), second.PWM.value=c(second.PWM.type.AUROC.numeric, second.PWM.type.AUPRC.numeric), AUC.type=c("AUROC", "AUPRC") ))
    })

    cross.comparison.data.frame[, "name"] <- temp.data.frame[1, "name"]
    cross.comparison.data.frame[, "accession"] <- temp.data.frame[1, "accession"]
    cross.comparison.data.frame[, "original.kernel.auc"] <- temp.data.frame[temp.data.frame[, "PWM.type"] == "original.kernel", "auc"]
    return(cross.comparison.data.frame)
##    temp.data.frame[, "Deepbind.AUC"] <- temp.data.frame[temp.data.frame[, "PWM.type"] == "Deepbind", , drop=FALSE][1, "AUC"]
##    temp.data.frame[, "original.kernel.AUC"] <- temp.data.frame[temp.data.frame[, "PWM.type"] == "original.kernel", , drop=FALSE][1, "AUC"]
##    temp.data.frame[, "AUC.relative.to.Deepbind"] <- temp.data.frame[, "AUC"] - temp.data.frame[, "Deepbind.AUC"]
##    temp.data.frame[, "AUC.relative.to.original.kernel"] <- temp.data.frame[, "AUC"] - temp.data.frame[, "original.kernel.AUC"]
##    return(temp.data.frame)
}, .progress="text")


write.csv(AUC.summary.paired.data.frame, "../data/for.2..9.2.AUC.summary.paired.txt")
# AUC.summary.paired.data.frame <- read.csv("../data/for.2..9.2.AUC.summary.paired.txt")
#ggplot(AUC.summary.with.Deepbind.as.baseline.data.frame, aes(x=PWM.type, y=AUC.relative.to.Deepbind)) + geom_boxplot()

png("../data/for.2..9.2.AUROC.and.AUPRC.difference.scatterplot.2.png", width=1000, height=600)
print(ggplot(subset(AUC.summary.paired.data.frame, first.PWM.type=="Deepbind" & second.PWM.type == "exp_enlarged_by_1.0"), aes(x=first.PWM.value, y=second.PWM.value)) + geom_point() + geom_abline(slope=1, intercept=0, color="red")  + xlim(0.5, 1) + ylim(0.5, 1) + scale_color_gradientn(colors=rainbow(10)[1:8], limits=c(0.5, 1)) + theme(text=element_text(size=28)) + facet_wrap(~AUC.type) + labs(x = "AUC of Deepbind's transformation", y = "AUC of the exact transformation",  color = "AUC of the original \n model (with kernels)"))
dev.off()


png("../data/for.2..9.2.AUROC.and.AUPRC.difference.scatterplot.2.histogram.png", width=1000, height=600)
print(ggplot(subset(AUC.summary.paired.data.frame, first.PWM.type=="Deepbind" & second.PWM.type == "exp_enlarged_by_1.0"), aes(x=second.PWM.value - first.PWM.value)) + geom_histogram() + geom_vline(xintercept=0, color="red") + theme(text=element_text(size=28)) + facet_wrap(~AUC.type) + labs(x = "AUC of the exact transformation minus\n AUC of Deepbind's transformation", y = "count"))
dev.off()



for.AUPRC.statistical.test.data.frame <- subset(AUC.summary.paired.data.frame, first.PWM.type=="Deepbind" & second.PWM.type == "exp_enlarged_by_1.0" & AUC.type == "AUPRC")
for.AUROC.statistical.test.data.frame <- subset(AUC.summary.paired.data.frame, first.PWM.type=="Deepbind" & second.PWM.type == "exp_enlarged_by_1.0" & AUC.type == "AUROC")

wilcox.test(for.AUPRC.statistical.test.data.frame[, "second.PWM.value"], for.AUPRC.statistical.test.data.frame[, "first.PWM.value"], alternative="greater", paired = TRUE)
wilcox.test(for.AUROC.statistical.test.data.frame[, "second.PWM.value"], for.AUROC.statistical.test.data.frame[, "first.PWM.value"], alternative="greater", paired = TRUE)

## to.plot.in.histogram.data.frame <- subset(AUC.summary.paired.data.frame, first.PWM.type=="Deepbind" & second.PWM.type == "exp_enlarged_by_1.0")

## to.plot.in.histogram.with.difference.data.frame <- adply(.data=to.plot.in.histogram.data.frame, .margins=1, .fun=function(row.values.data.frame){
##     row.values.data.frame[, "difference"] <- row.values.data.frame[1, "second.PWM.type.AUC"] - row.values.data.frame[1, "first.PWM.type.AUC"]
##     row.values.data.frame[, "difference.clipped"] <- max(min(row.values.data.frame[, "difference"], 0.1), -0.015)
##     return(row.values.data.frame)
## })

## png("../data/for.2..9.AUC.difference.plot.2.in.histogram.png", width=800, height=600)
## print( ggplot(to.plot.in.histogram.with.difference.data.frame, aes(x=difference.clipped, fill=factor(floor(original.kernel.auc*20)/20)) ) + geom_histogram(binwidth=0.001) + geom_segment(x=0, y=0, xend=0, yend=Inf, color="black", size=2) + scale_fill_manual(values=rainbow(14)[1:11]) + theme(text=element_text(size=28)) + labs(x = "AUC of our transformation\n - AUC of Deepbind's transformation,\n restricted by [-0.015, 0.1]", fill = "AUC of the original \n model (with kernels)"))
## dev.off()


## png("../data/for.2..9.AUC.difference.plot.2.in.histogram.my.only.png", width=800, height=600)
## print( ggplot(to.plot.in.histogram.with.difference.data.frame, aes(x=difference.clipped, fill=factor(floor(original.kernel.auc*20)/20)) ) + geom_histogram(binwidth=0.001) + geom_segment(x=0, y=0, xend=0, yend=Inf, color="black", size=2) + scale_fill_manual(values=rainbow(14)[1:11]) + theme(text=element_text(size=28)) + labs(x = "AUC of my transformation\n - AUC of Deepbind's transformation,\n restricted by [-0.015, 0.1]", fill = "AUC of the original \n model (with kernels)"))
## dev.off()
