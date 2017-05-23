/*
  This file contains function prototypes and constant definitions for EVSL
*/

#ifndef PEVSL_H
#define PEVSL_H

#include "pevsl_struct.h"
#include "pevsl_def.h"

#include "pevsl_protos.h"

int pEVSL_Start(int argc, char **argv);

int pEVSL_Finish();

int pEVSL_SetAParcsr(pevsl_Parcsr *A);


#endif

