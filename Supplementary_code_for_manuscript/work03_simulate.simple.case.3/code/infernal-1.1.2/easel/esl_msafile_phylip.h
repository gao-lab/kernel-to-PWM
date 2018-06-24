/* I/O of multiple sequence alignments in PHYLIP format
 */
#ifndef eslMSAFILE_PHYLIP_INCLUDED
#define eslMSAFILE_PHYLIP_INCLUDED

#include "esl_msa.h"
#include "esl_msafile.h"

extern int esl_msafile_phylip_SetInmap     (ESL_MSAFILE *afp);
extern int esl_msafile_phylip_GuessAlphabet(ESL_MSAFILE *afp, int *ret_type);
extern int esl_msafile_phylip_Read         (ESL_MSAFILE *afp, ESL_MSA **ret_msa);
extern int esl_msafile_phylip_Write        (FILE *fp, const ESL_MSA *msa, int format, ESL_MSAFILE_FMTDATA *opt_fmtd);

extern int esl_msafile_phylip_CheckFileFormat(ESL_BUFFER *bf, int *ret_format, int *ret_namewidth);

#endif /* eslMSAFILE_PHYLIP_INCLUDED */
/*****************************************************************
 * Easel - a library of C functions for biological sequence analysis
 * Version 0.43; July 2016
 * Copyright (C) 2016 Howard Hughes Medical Institute
 * Other copyrights also apply. See the LICENSE file for a full list.
 * 
 * Easel is open source software, distributed under the BSD license. See
 * the LICENSE file for more details.
 *****************************************************************/
