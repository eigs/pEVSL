#include "pevsl_int.h"

/*!
 * @brief Create a parallel multi-vector struct without allocating memory for data 
 */
void pEVSL_ParvecsCreateShell(int nglobal, int nvecs, int ld, int nlocal, int nfirst, 
                              MPI_Comm comm, pevsl_Parvecs *x, double *data) {
  x->comm = comm;
  x->n_global = nglobal;
  x->n_local = nlocal;
  x->n_first = nfirst < 0 ? PEVSL_NOT_DEFINED : nfirst;
  x->n_vecs = nvecs;
  x->ld = ld;
  PEVSL_CHKERR(ld < nlocal);
  x->data = data;
}

/*!
 * @brief Create a parallel multi-vector struct
 */
void pEVSL_ParvecsCreate(int nglobal, int nvecs, int ld, int nlocal, int nfirst,
                         MPI_Comm comm, pevsl_Parvecs *x) {
  
  pEVSL_ParvecsCreateShell(nglobal, nvecs, ld, nlocal, nfirst, comm, x, NULL);
  
  /* to prevent integer overflow, cast it to size_t before multiplication */
  size_t alloc = ((size_t) ld) * ((size_t) nvecs);
  PEVSL_MALLOC(x->data, alloc, double);
}

/*!
 * @brief Creates multi-vectors of the same type as an existing vector
 */
void pEVSL_ParvecsDuplFromParvec(pevsl_Parvec *x, int nvecs, int ld, pevsl_Parvecs *y) {

  pEVSL_ParvecsCreate(x->n_global, nvecs, ld, x->n_local, x->n_first, x->comm, y);
}

/*!
 * @brief Destroy a Parvecs struct
 */
void pEVSL_ParvecsFree(pevsl_Parvecs *x) {

  PEVSL_FREE(x->data);
}

