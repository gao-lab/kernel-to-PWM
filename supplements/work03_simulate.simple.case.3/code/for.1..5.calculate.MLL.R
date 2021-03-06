library("plyr")
library("stringr")
library("rhdf5")
library("Biostrings")
library("doMC")

registerDoMC(cores=20)

motif.name.character.vector <- readLines("../data/for.1..2.all.PWMs.name.txt")
dataset.name.character.vector <- c("training", "validation")

settings.data.frame <- expand.grid(motif.name=motif.name.character.vector, dataset.name=dataset.name.character.vector)

not.used.variable.1 <- a_ply(settings.data.frame, .margins=1, .fun=function(row.values.data.frame){

    cat(date(), " : Processing ", rownames(row.values.data.frame), " -- ", apply(row.values.data.frame, MARGIN=1, FUN=paste, collapse=", "), "\n")

    
    motif.name.character <- row.values.data.frame[1, "motif.name"]
    dataset.name.character <- row.values.data.frame[1, "dataset.name"]

    sequence.fasta.filename.character <- paste(sep="", "../data/for.1..2.sequence.dataset.__", dataset.name.character, "__", motif.name.character, ".fasta")
    meme.filename.character <- paste(sep="", "../data/for.1..4.meme.for.", motif.name.character, ".meme.txt")

    meme.lines.character.vector <- readLines(meme.filename.character)

    meme.motif.start.index.integer.vector <- grep(pattern="^MOTIF", x=meme.lines.character.vector)
    meme.motif.end.index.integer.vector <- c(meme.motif.start.index.integer.vector[-1] - 1 , length(meme.lines.character.vector))

    meme.motif.start.and.end.index.data.frame <- data.frame(start=meme.motif.start.index.integer.vector, end=meme.motif.end.index.integer.vector)

    motif.numeric.name.and.matrix.list.list <- alply(meme.motif.start.and.end.index.data.frame, .margins=1, .fun=function(row.value.data.frame){
        motif.lines.character.vector <- meme.lines.character.vector[ row.value.data.frame[1, "start"] : row.value.data.frame[1, "end"] ]
        motif.name.character <- sub(pattern="^MOTIF ([^ ]*)($| ).*", replacement="\\1", x=str_trim(motif.lines.character.vector[1]))
        motif.matrix.lines.character.vector <- grep(pattern="\t", x=motif.lines.character.vector, value=TRUE)
        motif.numeric.matrix <- do.call(cbind, lapply(str_split(string=str_trim(motif.matrix.lines.character.vector), pattern="[ \t]+"), as.numeric) )

        ## correction for 0
        motif.numeric.matrix[motif.numeric.matrix == 0] <- min(motif.numeric.matrix[motif.numeric.matrix != 0]) / 10
        
        motif.log.numeric.matrix <- log(motif.numeric.matrix)
        
        return(list(motif.name=motif.name.character, motif.matrix=motif.numeric.matrix, motif.log.matrix=motif.log.numeric.matrix, motif.lines.character.vector=motif.lines.character.vector))
    })

    names(motif.numeric.name.and.matrix.list.list) <- sapply(motif.numeric.name.and.matrix.list.list, function(motif.numeric.name.and.matrix.list){return(motif.numeric.name.and.matrix.list[["motif.name"]])})




    sequence.fasta.DNAStringSet <- readDNAStringSet(sequence.fasta.filename.character)

    sequence.matrix.list <- llply(names(sequence.fasta.DNAStringSet), .fun=function(single.sequence.name.character){
        single.sequence.character <- as.character(sequence.fasta.DNAStringSet[[single.sequence.name.character]])
        single.sequence.nucleotide.character.vector <- strsplit(x=single.sequence.character, split="")[[1]]
        single.sequence.length.integer <- length(single.sequence.nucleotide.character.vector)
        single.sequence.index.integer.vector <- c('A'=1, 'C'=2, 'G'=3, 'T'=4)[single.sequence.nucleotide.character.vector]
        single.sequence.integer.matrix <- do.call(cbind, lapply(1:single.sequence.length.integer, FUN=function(temp.index.integer){
            temp.column.integer.vector <- c(0, 0, 0, 0)
            temp.column.integer.vector[ single.sequence.index.integer.vector[temp.index.integer] ] <- 1
            return(temp.column.integer.vector)
        }))
        return(single.sequence.integer.matrix)
    })

    names(sequence.matrix.list) <- names(sequence.fasta.DNAStringSet)


    motif.and.sequence.pair.data.frame <- expand.grid(motif=names(motif.numeric.name.and.matrix.list.list), sequence=names(sequence.matrix.list), stringsAsFactors=FALSE)

    MLL.position.and.LL.across.all.motifs.and.all.sequences.data.frame <- adply(1:nrow(motif.and.sequence.pair.data.frame), .margins=1, .fun=function(temp.row.index.integer){

        
        row.values.data.frame <- motif.and.sequence.pair.data.frame[temp.row.index.integer, , drop=FALSE]
        motif.name.character <- row.values.data.frame[1, "motif"]
        sequence.name.character <- row.values.data.frame[1, "sequence"]        
        motif.log.numeric.matrix <- motif.numeric.name.and.matrix.list.list[[ motif.name.character  ]][[ "motif.log.matrix"  ]]
        sequence.integer.matrix <- sequence.matrix.list[[ sequence.name.character ]]

        motif.length.integer <- ncol(motif.log.numeric.matrix)
        sequence.length.integer <- ncol(sequence.integer.matrix)
        position.and.log.likelihood.data.frame <- ldply((1:(sequence.length.integer - motif.length.integer + 1)), .fun=function(temp.index.integer){
            return(c(position=temp.index.integer, score=sum(sequence.integer.matrix[,  temp.index.integer:(temp.index.integer + motif.length.integer - 1)  ] * motif.log.numeric.matrix)))
        })
        MLL.position.and.LL.data.frame <- position.and.log.likelihood.data.frame[which.max(position.and.log.likelihood.data.frame[, "score"]), , drop=FALSE]
        MLL.position.and.LL.data.frame[, "PWM.name"] <- motif.name.character
        MLL.position.and.LL.data.frame[, "sequence.name"] <- sequence.name.character
        return(MLL.position.and.LL.data.frame)
    })


    maximal.log.likelihood.output.filename.character <- paste(sep="", "../data/for.1..5.MLL.for.motif.", motif.name.character , ".and.dataset.", dataset.name.character, ".fimo.txt")

    write.table(x=MLL.position.and.LL.across.all.motifs.and.all.sequences.data.frame[, c("PWM.name", "sequence.name", "score")], file=maximal.log.likelihood.output.filename.character, sep="\t", row.names=FALSE, col.names=TRUE, quote=FALSE )

}, .parallel=TRUE)
