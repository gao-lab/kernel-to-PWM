library("plyr")

sequence.length.integer <- 5000
cm.motif.name.and.accession.data.frame <- read.table(file="../data/for.2..1.Rfam.motif.name.and.accession.txt", header=TRUE, sep="\t", stringsAsFactors=FALSE)



PWM.type.and.auc.augmented.with.original.kernel.model.across.all.CMs.data.frame <- adply(cm.motif.name.and.accession.data.frame, .margins=1, .fun=function(row.values.data.frame){

    CM.name.character <- row.values.data.frame[1, "name"]
    CM.ACC.character <- row.values.data.frame[1, "accession"]

    CM.dir.filename.character <- paste(sep="", "../data/for.2..7.simulation.", "CM.accession__", CM.ACC.character, ".CM.name__", CM.name.character,  ".sequence.length__", sequence.length.integer, "/")

    if (file.exists(paste(sep="", CM.dir.filename.character, "for.2..7.8.PWM.type.and.auroc.and.auprc.txt")) == FALSE){
        cat(date(), " : skipping CM ",  CM.name.character, " / ", CM.ACC.character , "\n")
        return(NULL)
    }

    original.kernel.model.auc.data.frame <- read.table(paste(sep="", CM.dir.filename.character, "for.2..7.3.auc.txt"), header=FALSE, sep="\t", stringsAsFactors=FALSE)
    original.kernel.training.and.validation.auc.numeric <- original.kernel.model.auc.data.frame[1, 3]

    PWM.type.and.auc.data.frame <- read.table(paste(sep="", CM.dir.filename.character, "for.2..7.8.PWM.type.and.auroc.and.auprc.txt"), header=TRUE, sep="\t", stringsAsFactors=FALSE)

    PWM.type.and.auc.augmented.with.original.kernel.model.data.frame <- cbind(PWM.type.and.auc.data.frame, data.frame(PWM.type="original.kernel", auc=original.kernel.training.and.validation.auc.numeric, stringsAsFactors=FALSE))
    PWM.type.and.auc.augmented.with.original.kernel.model.data.frame[, "CM.name"] <- CM.name.character
    PWM.type.and.auc.augmented.with.original.kernel.model.data.frame[, "CM.accession"] <- CM.ACC.character
    return(PWM.type.and.auc.augmented.with.original.kernel.model.data.frame)
})


write.table(PWM.type.and.auc.augmented.with.original.kernel.model.across.all.CMs.data.frame, file="../data/for.2..8.2.AUROC.and.AUPRC.summary.for.Rfam.txt", col.names=TRUE, row.names=FALSE, sep="\t", quote=FALSE)
