library("plyr")

deepbind.motif.meta.data.frame <- read.table(file="../data/for.0..2.deepbind-v0.11-linux/db/db.tsv", sep="\t", header=TRUE, stringsAsFactors=FALSE)




## PBM
## don't know how to parse (no sequence ~ motif+experiment mapping)
## E:/Results/deepbind/dream5/final/C_1 -> dream5/pbm/


## RNAcompete
## A1CF + E:/Results/deepbind/rnac/AB/final/RNCMPT00001 -> rnac/invivo/A1CF.txt (this file does not exist)
## need to check whether the source file exists
## Also, note that this file has a format different from those of ChIP-seq and SELEX


## Conclusion: run all motifs of ChIP-Seq and SELEX first


deepbind.motif.meta.valid.data.frame <- subset(deepbind.motif.meta.data.frame, Labels != "deprecated" & Experiment %in% c("ChIP-Seq", "SELEX"))

input.command.line.argument.data.frame <- adply(.data=deepbind.motif.meta.valid.data.frame, .margins=1, .fun=function(row.values.data.frame){
    motif.ID.character <- row.values.data.frame[1, "ID"]
    motif.experiment.character <- row.values.data.frame[1, "Experiment"]
    motif.path.character <- row.values.data.frame[1, "Path"]

    motif.dataset.filename.character <- NULL
    if (motif.experiment.character == "ChIP-seq"){
        ## ChIP-seq
        ## E:/Results/deepbind/encode/best/final/ARID3A_K562_ARID3A_(sc-8821)_Stanford -> /encode/ARID3A_K562_ARID3A_(sc-8821)_Stanford_AC.seq.gz
        motif.dataset.filename.character <- sub(pattern="^.*encode/best/final/(.*)$", replacement="/encode/\\1_AC.seq.gz", x=motif.path.character)
    } else if (motif.experiment.character == "SELEX"){
        ## SELEX
        ## E:/Results/deepbind/selex/best/final/Alx1_DBD_TAAAGC20NCG_3_Z -> selex/best/Alx1_DBD_TAAAGC20NCG_3_Z_A.seq.gz
        motif.dataset.filename.character <- sub(pattern="^.*selex/best/final/(.*)$", replacement="/selex/best/\\1_A.seq.gz", x=motif.path.character)
    }
    return(data.frame(ID=motif.ID.character, dataset.filename=motif.dataset.filename.character, stringsAsFactors=FALSE))
})[, c("ID", "dataset.filename")]

write.table(x=input.command.line.argument.data.frame, file="../data/for.2..1.input.command.line.argument.txt", sep="\t", row.names=FALSE, col.names=TRUE, quote=FALSE)
    ## TODO
## Remove all average-pooling motifs
