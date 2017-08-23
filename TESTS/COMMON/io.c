#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"

// parse command-line input parameters
int findarg(const char *argname, ARG_TYPE type, void *val, int argc, char **argv) {
  int *outint;
  double *outdouble;
  char *outchar;
  int i;
  for (i=0; i<argc; i++) {
    if (argv[i][0] != '-') {
      continue;
    }
    if (!strcmp(argname, argv[i]+1)) {
      if (type == NA) {
        return 1;
      } else {
        if (i+1 >= argc /*|| argv[i+1][0] == '-'*/) {
          return 0;
        }
        switch (type) {
          case INT:
            outint = (int *) val;
            *outint = atoi(argv[i+1]);
            return 1;
            break;
          case DOUBLE:
            outdouble = (double *) val;
            *outdouble = atof(argv[i+1]);
            return 1;
            break;
          case STR:
            outchar = (char *) val;
            sprintf(outchar, "%s", argv[i+1]);
            return 1;
            break;
          default:
            printf("unknown arg type\n");
        }
      }
    }
  }
  return 0;
}

/* job = 0  : open matfile, read the number of problems to test, and save the number in io
 * job = 1  : read one line for one problem [standard e.v problem]
 * job = 11 : read one line for one problem [generalized e.v problem]
 * job = 2  : close the matfile
 */
int GetMatfileInfo(io_t *io, int job, const char *fname) {

  char path1[MAX_LINE], path2[MAX_LINE], MatNam1[MAX_LINE], MatNam2[MAX_LINE], 
       Fmt[MAX_LINE], ca[MAX_LINE], cb[MAX_LINE], cn_intv[MAX_LINE];
  FILE *fmat;

  /* open the file and read num of tests */
  if (job == 0) {
    if ( NULL == (fmat = fopen(fname, "r")) ) {
      printf("Can't open file %s...\n", fname);
      return -1;
    }
    /* get the number of test problems */
    if (1 != fscanf(fmat, "%d\n", &io->ntests)) {
      printf("Error in reading num of test probs...\n" );
      return 1;
    }
    io->fin = fmat;

    return 0;
  }
  
  /* close file */
  if (job == 2) {
    fclose(io->fin);
    return 0;
  }
 
  if (job == 1) {
    fmat = io->fin;
    /*-------------------- READ LINE */
    if (6 != fscanf(fmat,"%s %s %s %s %s %s\n", path1, MatNam1, Fmt, ca, cb, cn_intv) ) {
      printf("Error in reading a problem line...\n");
      return 2;
    }
  } else if (job == 11) {
    fmat = io->fin;
    /*-------------------- READ LINE */
    if (8 != fscanf(fmat,"%s %s %s %s %s %s %s %s\n", path1, path2, MatNam1, MatNam2, 
                    Fmt, ca, cb, cn_intv) ) {
      printf("Error in reading a problem line...\n");
      return 2;
    }
  }
  
  /*-------------------- matrix format */
  if (strcmp(Fmt,"HB") == 0) {
    io->Fmt = HB;
    sprintf(io->FmtStr, "%s", "HB");
    printf("HB matrix format has not yet been supported...\n");
    return 4;
  } else if (strcmp(Fmt,"MM0") == 0) {
    io->Fmt = MM0;
    sprintf(io->FmtStr, "%s", "MM0");
  } else if (strcmp(Fmt,"MM1") == 0) {
    io->Fmt = MM1;
    sprintf(io->FmtStr, "%s", "MM1");
  } else {
    /*-------------------- UNKNOWN_FORMAT */
    io->Fmt = UNKNOWN_FORMAT;
    sprintf(io->FmtStr, "%s", "UNKNOWN");
    printf("Error unknown matrix format ...\n");
    return 3;
  }

  strcpy(io->Fname1, path1);
  strcpy(io->MatNam1, MatNam1);
  if (job == 11) {
    strcpy(io->Fname2, path2);
    strcpy(io->MatNam2, MatNam2);
  }

  io->a = atof(ca);
  io->b = atof(cb);
  io->nslices = atoi(cn_intv);
  
  /*
  if (job == 1) {
    printf(" Problem: A: %s(%s), Format: %s, intv [%e, %e], nslices %d \n", io->MatNam1, 
           io->Fname1, Fmt, io->a, io->b, io->nslices);
  } else if (job == 11) {
    printf(" Problem: A: %s(%s), B: %s(%s), Format: %s, intv [%e, %e], nslices %d \n", io->MatNam1, 
           io->Fname1, io->MatNam2, io->Fname2, Fmt, io->a, io->b, io->nslices);
  }
  */

  return 0;
}

