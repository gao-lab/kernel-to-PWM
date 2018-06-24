export ${PYTHON_EXEC}=python
export ${RSCRIPT_EXEC}=Rscript

## for.1..2.0.generate.input.for.step.7.with.smaller.base.py
## this step generates the following files for each motif:

${PYTHON_EXEC} ./for.1..2.0.generate.input.for.step.7.with.smaller.base.py "D00600.001" "/selex/best/PRDM1_FL_TTGAGG20NGAT_2_AE_A.seq.gz"
${PYTHON_EXEC} ./for.1..2.0.generate.input.for.step.7.with.smaller.base.py "D00600.004" /encode/PRDM1_HeLa-S3_PRDM1_\(9115\)_Stanford_AC.seq.gz
${PYTHON_EXEC} ./for.1..2.0.generate.input.for.step.7.with.smaller.base.py "D00350.002" "/selex/best/EBF1_FL_TATAAG20NCG_3_AC_A.seq.gz"
${PYTHON_EXEC} ./for.1..2.0.generate.input.for.step.7.with.smaller.base.py "D00350.005" /encode/EBF1_GM12878_EBF1_\(SC-137065\)_Stanford_AC.seq.gz
${PYTHON_EXEC} ./for.1..2.0.generate.input.for.step.7.with.smaller.base.py "D00558.002" "/selex/best/NR4A2_FL_TCATTG20NTTA_3_W_A.seq.gz"
${PYTHON_EXEC} ./for.1..2.0.generate.input.for.step.7.with.smaller.base.py "D00328.003" "/selex/best/CTCF_FL_TAGCGA20NGCT_4_AJ_A.seq.gz"
${PYTHON_EXEC} ./for.1..2.0.generate.input.for.step.7.with.smaller.base.py "D00694.003" "/selex/best/Tp53_DBD_TCGGGG20NGGT_4_AK_A.seq.gz"
${PYTHON_EXEC} ./for.1..2.0.generate.input.for.step.7.with.smaller.base.py "D00588.002" "/selex/best/Pou2f2_DBD_TAGATA20NTAT_3_Z_A.seq.gz"

## for.1..2.1.run.step.7.in.parallel.with.smaller.base.py

for motif_id in "D00600.001" "D00600.004" "D00350.002" "D00350.005" "D00558.002" "D00328.003" "D00694.003" "D00588.002"
do
    ${PYTHON_EXEC} ./for.1..2.1.run.step.7.in.parallel.with.smaller.base.py ${motif_id}
done

## for.1..2.3.generate.real.dataset.with.smaller.base.py

${PYTHON_EXEC} ./for.1..2.3.generate.real.dataset.with.smaller.base.py "D00600.001" "/selex/best/PRDM1_FL_TTGAGG20NGAT_2_AE_A.seq.gz"
${PYTHON_EXEC} ./for.1..2.3.generate.real.dataset.with.smaller.base.py "D00600.004" /encode/PRDM1_HeLa-S3_PRDM1_\(9115\)_Stanford_AC.seq.gz
${PYTHON_EXEC} ./for.1..2.3.generate.real.dataset.with.smaller.base.py "D00350.002" "/selex/best/EBF1_FL_TATAAG20NCG_3_AC_A.seq.gz"
${PYTHON_EXEC} ./for.1..2.3.generate.real.dataset.with.smaller.base.py "D00350.005" /encode/EBF1_GM12878_EBF1_\(SC-137065\)_Stanford_AC.seq.gz
${PYTHON_EXEC} ./for.1..2.3.generate.real.dataset.with.smaller.base.py "D00558.002" "/selex/best/NR4A2_FL_TCATTG20NTTA_3_W_A.seq.gz"
${PYTHON_EXEC} ./for.1..2.3.generate.real.dataset.with.smaller.base.py "D00328.003" "/selex/best/CTCF_FL_TAGCGA20NGCT_4_AJ_A.seq.gz"
${PYTHON_EXEC} ./for.1..2.3.generate.real.dataset.with.smaller.base.py "D00694.003" "/selex/best/Tp53_DBD_TCGGGG20NGGT_4_AK_A.seq.gz"
${PYTHON_EXEC} ./for.1..2.3.generate.real.dataset.with.smaller.base.py "D00588.002" "/selex/best/Pou2f2_DBD_TAGATA20NTAT_3_Z_A.seq.gz"

## for.1..2.4.run.the.rest.with.smaller.base.py

for motif_id in "D00600.001" "D00600.004" "D00350.002" "D00350.005" "D00558.002" "D00328.003" "D00694.003" "D00588.002"
do
    time ${PYTHON_EXEC} ./for.1..2.4.run.the.rest.with.smaller.base.py ${motif_id}
done

## for.1..2.6.summarize.result.R

${RSCRIPT_EXEC} ./for.1..2.6.summarize.result.R
