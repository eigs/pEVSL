#include "pevsl_protos.h"

void pEVSL_ParvecCreate(int nglobal, int nlocal, int nfirst, MPI_Comm comm, pevsl_Parvec *x) {
  x->comm = comm;
  x->n_global = nglobal;
  x->n_local = nlocal;
  //x->n_first = nfirst;
  PEVSL_MALLOC(x->data, nlocal, double);
}


void pEVSL_ParvecDupl(pevsl_Parvec *x, pevsl_Parvec *y) {
  y->comm = x->comm;
  y->n_global = x->n_global;
  y->n_local = x->n_local;
  //y->n_first = x->n_first;    
  PEVSL_MALLOC(y->data, y->n_local, double);
}

void pEVSL_ParvecFree(pevsl_Parvec *x) {
  PEVSL_FREE(x->data);
}

void pEVSL_ParvecRand(pevsl_Parvec *x) {
  int i;
  double t = ((double) RAND_MAX) / 2.0;
  /*
     for (i=0; i<x->n_global; i++) {
     double z = (rand() - t) / t;
     if (i >= x->n_first && i < x->n_first + x->n_local) {
     x->data[i-x->n_first] = z;
     }
     }
     */
  for (i=0; i<x->n_local; i++) {
    double z = (rand() - t) / t;
    x->data[i] = z;
  }
}

void pEVSL_ParvecDot(pevsl_Parvec *x, pevsl_Parvec *y, double *t) {
  PEVSL_CHKERR(x->n_global != y->n_global);
  PEVSL_CHKERR(x->n_local != y->n_local);
  //PEVSL_CHKERR(x->n_first != y->n_first);
  double tlocal;
  int one = 1;
  tlocal = DDOT(&(x->n_local), x->data, &one, y->data, &one);
  MPI_Allreduce(&tlocal, t, 1, MPI_DOUBLE, MPI_SUM, x->comm);
}

void pEVSL_ParvecNrm2(pevsl_Parvec *x, double *t) {
  double t2;
  pEVSL_ParvecDot(x, x, &t2);
  *t = sqrt(t2);
}

// y := x
void pEVSL_ParvecCopy(pevsl_Parvec *x, pevsl_Parvec *y) {
  PEVSL_CHKERR(x->n_global != y->n_global);
  PEVSL_CHKERR(x->n_local != y->n_local);
  //PEVSL_CHKERR(x->n_first != y->n_first);
  int one = 1;
  DCOPY(&(x->n_local), x->data, &one, y->data, &one);
}

void pEVSL_ParvecSum(pevsl_Parvec *x, double *t) {
  double localsum = 0.0;
  int i;
  for (i=0; i<x->n_local; i++) {
    localsum += x->data[i];
  }
  MPI_Allreduce(&localsum, t, 1, MPI_DOUBLE, MPI_SUM, x->comm);
}

void pEVSL_ParvecScal(pevsl_Parvec *x, double t) {
  int one = 1;
  DSCAL(&(x->n_local), &t, x->data, &one);
}

/*
void pEVSL_ParvecAddScalar(pevsl_Parvec *x, double t) {
  int i;
  for (i=0; i<x->n_local; i++) {
    x->data[i] += t;
  }
}

void pEVSL_ParvecSetScalar(pevsl_Parvec *x, double t) {
  int i;
  for (i=0; i<x->n_local; i++) {
    x->data[i] = t;
  }
}

*/

void pEVSL_ParvecSetZero(pevsl_Parvec *x) {
  memset(x->data, 0, x->n_local*sizeof(double));
}

void pEVSL_ParvecAxpy(double a, pevsl_Parvec *x, pevsl_Parvec *y) {
  PEVSL_CHKERR(x->n_global != y->n_global);
  PEVSL_CHKERR(x->n_local != y->n_local);
  //PEVSL_CHKERR(x->n_first != y->n_first);
  int one = 1;
  DAXPY(&(x->n_local), &a, x->data, &one, y->data, &one);
}


int pEVSL_ParvecSameSize(pevsl_Parvec *x, pevsl_Parvec *y) {
  if (x->n_global != y->n_global ||
      x->n_local != y->n_local) /*||
      x->n_first != y->n_first)*/ {
    return 0;
  }
  return 1;
}

/** @brief print Parvec on comm (squential in comm) 
 *  @param fn: filename (if NULL, print it to stdout) */
int pEVSL_ParvecWrite(pevsl_Parvec *x, const char *fn, MPI_Comm comm) {
  /* if fn == NULL, print to stdout */
  FILE *fp = fn ? NULL : stdout;
  /* sequential loop with all ranks in comm */
  PEVSL_SEQ_BEGIN(comm, rank, size)
  {
    /* if print to file, open it
     * mode 'a' should work */
    if (fn) {
      fp = fopen(fn, "a");
    }
    /* if fp != NULL */
    if (fp) {
      /* rank 0 writes the global size */
      if (rank == 0) {
        fprintf (fp, "%% %d\n", x->n_global);
      }
      /* write local data */
      int j;
      for (j=0; j<x->n_local; j++) {
        fprintf(fp, "%.15e\n", x->data[j]);
      }
      if (fn) {
        fclose(fp);
      }
    }
  }
  PEVSL_SEQ_END(comm)

  return 0;
}

/*
   int pEVSL_ParvecRead(pevsl_Parvec *x, const char* fn) {
   char str[PEVSL_MAX_LINE];
   int k;
   FILE *fp = fopen(fn, "r");
   do {
   fgets(str, PEVSL_MAX_LINE, fp);
   } while (str[0] == '%');

   for (k=0; k<x->n_first+x->n_local; k++) {
   if (k >= x->n_first) {
   x->data[k-x->n_first] = atof(str);
   }
   fscanf(fp, "%s\n", str);
   }
   fclose(fp);

   return 0;
   }
   */
