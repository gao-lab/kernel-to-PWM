{RSCRIPT_EXEC}=Rscript

# for.2..1.generate.input.command.lines.R

${RSCRIPT_EXEC} ./for.2..1.generate.input.command.lines.R

# for.2..2.run.all.steps.for.a.single.motif.and.experiment.sh
cat ../data/for.2..1.input.command.line.argument.txt | tail -n +2 | \
    while read line
    do
        bash ./for.2..2.run.all.steps.for.a.single.motif.and.experiment.sh ${line}
    done

# for.2..3.summarize.results.R

${RSCRIPT_EXEC} ./for.2..3.summarize.results.R
