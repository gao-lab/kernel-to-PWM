/* PAML interface
 *
 *   "Phylogenetic Analysis by Maximum Likelihood"
 *   Ziheng Yang
 *   http://abacus.gene.ucl.ac.uk/software/paml.html
 *   [Yang97]
 * 
 *           incept: SRE, Tue Jul 13 13:20:08 2004 [St. Louis]
 * upgrade to Easel: SRE, Thu Mar  8 13:26:20 2007 [Janelia]
 * SVN $Id$
 * SVN $URL$
 */
#ifndef eslPAML_INCLUDED
#define eslPAML_INCLUDED

#include <stdio.h>
#include <esl_dmatrix.h>

extern int esl_paml_ReadE(FILE *fp, ESL_DMATRIX *E, double *pi);


#endif /*eslPAML_INCLUDED*/
/*****************************************************************
 * Easel - a library of C functions for biological sequence analysis
 * Version 0.43; July 2016
 * Copyright (C) 2016 Howard Hughes Medical Institute
 * Other copyrights also apply. See the LICENSE file for a full list.
 * 
 * Easel is open source software, distributed under the BSD license. See
 * the LICENSE file for more details.
 *****************************************************************/
