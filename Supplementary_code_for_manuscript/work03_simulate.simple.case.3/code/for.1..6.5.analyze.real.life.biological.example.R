first.and.second.PWM.auc.across.all.PWM.names.data.frame <- read.table("../data/for.1..6.4.first.and.second.PWM.auc.across.all.PWM.names.txt", header=TRUE)


subset(first.and.second.PWM.auc.across.all.PWM.names.data.frame, (first.PWM.type == "Deepbind") & (second.PWM.type == "exp") & first.PWM.value - second.PWM.value < -0.03)[, c("PWM.name", "first.PWM.value", "second.PWM.value", "AUC.type")]


library("plyr")
library("stringr")
library("seqLogo")

motif.name.character <- "MA0278.1"
meme.filename.character <- paste(sep="", "../data/for.1..4.meme.for.", motif.name.character, ".meme.txt")
meme.lines.character.vector <- readLines(meme.filename.character)

meme.motif.start.index.integer.vector <- grep(pattern="^MOTIF", x=meme.lines.character.vector)
meme.motif.end.index.integer.vector <- c(meme.motif.start.index.integer.vector[-1] - 1 , length(meme.lines.character.vector))

meme.motif.start.and.end.index.data.frame <- data.frame(start=meme.motif.start.index.integer.vector, end=meme.motif.end.index.integer.vector)

motif.numeric.name.and.matrix.list.list <- alply(meme.motif.start.and.end.index.data.frame, .margins=1, .fun=function(row.value.data.frame){
    motif.lines.character.vector <- meme.lines.character.vector[ row.value.data.frame[1, "start"] : row.value.data.frame[1, "end"] ]
    motif.name.character <- sub(pattern="^MOTIF ([^ ]*)($| ).*", replacement="\\1", x=str_trim(motif.lines.character.vector[1]))
    motif.matrix.lines.character.vector <- grep(pattern="^[0-9\\. \t]+$", x=motif.lines.character.vector, value=TRUE)
    motif.numeric.matrix <- do.call(cbind, lapply(str_split(string=str_trim(motif.matrix.lines.character.vector), pattern="[ \t]+"), as.numeric) )

    ## correction for 0
    motif.numeric.matrix[motif.numeric.matrix == 0] <- min(motif.numeric.matrix[motif.numeric.matrix != 0]) / 10

    ## motif.log.numeric.matrix <- log(motif.numeric.matrix)

    return(list(motif.name=motif.name.character, motif.matrix=motif.numeric.matrix))
    })

    names(motif.numeric.name.and.matrix.list.list) <- sapply(motif.numeric.name.and.matrix.list.list, function(motif.numeric.name.and.matrix.list){return(motif.numeric.name.and.matrix.list[["motif.name"]])})

motif.kernel.matrix.list <- lapply(motif.numeric.name.and.matrix.list.list, function(temp.result){
    kernel.matrix <- log(temp.result[["motif.matrix"]])
    kernel.differenced.matrix <- apply(kernel.matrix, MARGIN=2, FUN=function(temp.col){
        return(temp.col - temp.col[1])
    })
    return(kernel.differenced.matrix)
})

motif.kernel.matrix.list[["original_normalized"]][, 1:15]
motif.kernel.matrix.list[["Deepbind_0"]][, -1 * (1:6)]

## png("../data/for.1..6.5.test.exp.png", width=800, height=600)
## print(seqLogo(motif.numeric.name.and.matrix.list.list[["exp_0"]][["motif.matrix"]]))
## dev.off()

## png("../data/for.1..6.5.test.Deepbind.png", width=800, height=600)
## print(seqLogo(motif.numeric.name.and.matrix.list.list[["Deepbind_0"]][["motif.matrix"]]))
## dev.off()

## png("../data/for.1..6.5.test.original.png", width=800, height=600)
## print(seqLogo(motif.numeric.name.and.matrix.list.list[["original_normalized"]][["motif.matrix"]]))
## dev.off()
