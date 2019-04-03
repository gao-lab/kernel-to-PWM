RSCRIPT_EXEC=Rscript

mkdir -p ../data
mkdir -p ../external

# step 1. read known motifs

${RSCRIPT_EXEC} ./for.2..1.split.Rfam.into.individual.cm.files.R

# steps 2-7 (each loop is for a separate motif)

cat ../data/for.2..1.Rfam.motif.name.and.accession.txt | tail -n +2 | \
    while read line
    do
        echo `date` ${line}
        bash ./for.2..7.simulate.single.cm.sh ${line} 5000
    done

# step 8. get AUC for the re-trained models of the two PFMs

${RSCRIPT_EXEC} ./for.2..8.2.summarize.AUPRC.and.AUROC.difference.R

# step 9. compare the AUCs

${RSCRIPT_EXEC} ./for.2..9.2.plot.AUROC.and.AUPRC.difference.R
