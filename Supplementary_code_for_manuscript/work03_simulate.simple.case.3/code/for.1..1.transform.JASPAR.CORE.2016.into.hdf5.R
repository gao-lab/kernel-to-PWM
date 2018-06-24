library("rhdf5")
library("plyr")
library("stringr")

meme.filename.character <- "../data/for.0..1.JASPAR_CORE_2016.meme"
PWM.hdf5.filename.character <- "../data/for.1..1.JAPSAR.CORE.2016.PWM.hdf5"

meme.lines.character.vector <- readLines(meme.filename.character)

meme.motif.start.index.integer.vector <- grep(pattern="^MOTIF", x=meme.lines.character.vector)
meme.motif.end.index.integer.vector <- c(meme.motif.start.index.integer.vector[-1] - 1 , length(meme.lines.character.vector))

meme.motif.start.and.end.index.data.frame <- data.frame(start=meme.motif.start.index.integer.vector, end=meme.motif.end.index.integer.vector)

motif.numeric.name.and.matrix.list.list <- alply(meme.motif.start.and.end.index.data.frame, .margins=1, .fun=function(row.value.data.frame){
    motif.lines.character.vector <- meme.lines.character.vector[ row.value.data.frame[1, "start"] : row.value.data.frame[1, "end"] ]
    motif.name.character <- sub(pattern="^MOTIF ([^ ]*)($| ).*", replacement="\\1", x=str_trim(motif.lines.character.vector[1]))
    motif.matrix.lines.character.vector <- grep(pattern="^[0-9\\. \t]+$", x=motif.lines.character.vector, value=TRUE)
    motif.numeric.matrix <- do.call(rbind, lapply(str_split(string=str_trim(motif.matrix.lines.character.vector), pattern="[ \t]+"), as.numeric) )
    return(list(motif.name=motif.name.character, motif.matrix=motif.numeric.matrix, motif.lines.character.vector=motif.lines.character.vector))
}, .progress="text")


file.create(PWM.hdf5.filename.character)

not.used.variable.1 <- l_ply(motif.numeric.name.and.matrix.list.list, .fun=function(motif.numeric.name.and.matrix.list){
    h5write(obj=motif.numeric.name.and.matrix.list[["motif.matrix"]], file=PWM.hdf5.filename.character, name=motif.numeric.name.and.matrix.list[["motif.name"]])
    H5close()
}, .progress="text")




