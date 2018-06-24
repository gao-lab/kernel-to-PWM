library("plyr")


cm.filename.character <- "../data/for.0..2.Rfam.12.2.cm"

cm.lines.character.vector <- readLines(cm.filename.character)

cm.motif.start.index.integer.vector <- grep(pattern="^INFERNAL", x=cm.lines.character.vector)
cm.motif.end.index.integer.vector <- c(cm.motif.start.index.integer.vector[-1] - 1 , length(cm.lines.character.vector))
cm.motif.name.character.vector <- sub(pattern="^NAME[ \t]+", replacement="", x=cm.lines.character.vector[cm.motif.start.index.integer.vector + 1])
cm.motif.accession.character.vector <- sub(pattern="^ACC[ \t]+", replacement="", x=cm.lines.character.vector[cm.motif.start.index.integer.vector + 2])
#cm.motif.clen.integer.vector <- as.integer(sub(pattern="^CLEN[ \t]+", replacement="", x=cm.lines.character.vector[cm.motif.start.index.integer.vector + 5]))

cm.motif.start.and.end.and.name.and.accession.data.frame <- data.frame(start=cm.motif.start.index.integer.vector, end=cm.motif.end.index.integer.vector, name=cm.motif.name.character.vector, accession=cm.motif.accession.character.vector, stringsAsFactors=FALSE)

not.used.variable.1 <- a_ply(cm.motif.start.and.end.and.name.and.accession.data.frame, .margins=1, .fun=function(row.value.data.frame){
    motif.lines.character.vector <- cm.lines.character.vector[ row.value.data.frame[1, "start"] : row.value.data.frame[1, "end"] ]
    motif.name.character <- row.value.data.frame[1, "name"]
    motif.accession.character <- row.value.data.frame[1, "accession"]

    motif.output.filename.character <- paste(sep="", "../data/for.2..1.Rfam..ACC__", motif.accession.character, "..NAME__", motif.name.character, ".cm")

    writeLines(text=motif.lines.character.vector, con=motif.output.filename.character)
    return(NULL)
}, .progress="text")



cm.motif.name.and.accession.data.frame <- cm.motif.start.and.end.and.name.and.accession.data.frame[, c("name", "accession")]

write.table(x=cm.motif.name.and.accession.data.frame, file="../data/for.2..1.Rfam.motif.name.and.accession.txt", row.names=FALSE, col.names=TRUE, sep="\t", quote=FALSE)
