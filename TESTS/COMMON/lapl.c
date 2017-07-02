#include <float.h>
#include "pevsl.h"

int LocalLapGen(int nx, int ny, int nz, int m1, int m2, pevsl_Csr *csr) {
    pevsl_Coo coo;
    pevsl_Coo *Acoo = &coo;

    int nrow = m2 - m1;
    int ncol = nx * ny * nz;
    Acoo->nrows = nrow;
    Acoo->ncols = ncol;

    int nzmax = nz > 1 ? 7*nrow : 5*nrow;
    PEVSL_MALLOC(Acoo->ir, nzmax, int);
    PEVSL_MALLOC(Acoo->jc, nzmax, int);
    PEVSL_MALLOC(Acoo->vv, nzmax, double);

    int ii, nnz=0;
    for (ii=m1; ii<m2; ii++) {
        double v = -1.0;
        double d = nz > 1 ? 6.0 : 4.0;
        int i,j,k,jj,ii2;

        k = ii / (nx*ny);
        i = (ii - k*nx*ny) / nx;
        j = ii - k*nx*ny - i*nx;
        ii2 = ii - m1;

        if (k > 0) {
            jj = ii - nx * ny;
            Acoo->ir[nnz] = ii2;  Acoo->jc[nnz] = jj;  Acoo->vv[nnz] = v;  nnz++;
        }

        if (i > 0) {
            jj = ii - nx;
            Acoo->ir[nnz] = ii2;  Acoo->jc[nnz] = jj;  Acoo->vv[nnz] = v;  nnz++;
        }

        if (j > 0) {
            jj = ii - 1;
            Acoo->ir[nnz] = ii2;  Acoo->jc[nnz] = jj;  Acoo->vv[nnz] = v;  nnz++;
        }

        jj = ii;
        Acoo->ir[nnz] = ii2;  Acoo->jc[nnz] = jj;  Acoo->vv[nnz] = d;  nnz++;
        
        if (j < nx-1) {
            jj = ii + 1;
            Acoo->ir[nnz] = ii2;  Acoo->jc[nnz] = jj;  Acoo->vv[nnz] = v;  nnz++;
        }

        if (i < ny-1) {
            jj = ii + nx;
            Acoo->ir[nnz] = ii2;  Acoo->jc[nnz] = jj;  Acoo->vv[nnz] = v;  nnz++;
        }

        if (k < nz-1) {
            jj = ii + nx * ny;
            Acoo->ir[nnz] = ii2;  Acoo->jc[nnz] = jj;  Acoo->vv[nnz] = v;  nnz++;
        }
    }

    Acoo->nnz = nnz;

    int ierr = pEVSL_CooToCsr(0, Acoo, csr); PEVSL_CHKERR(ierr);

    pEVSL_FreeCoo(Acoo);

    return 0;
}

/* @brief create a parallel csr matrix of 3-D Laplacians
 * A         : parcsr matrix
 * nx, ny, nz: grid sizes
 * row_starts: row starts of each MPI rank (of size NP + 1)
 * col_starts: col starts of each MPI rank (of size NP + 1)
 * comm      : MPI communicator of A
 */ 
int ParcsrLaplace2(pevsl_Parcsr *A, int nx, int ny, int nz, int *row_starts, 
                   int *col_starts, MPI_Comm comm) {

  /* row range of this rank: [r1, r2) and
   * col range of this rank: [c1, c2) */
  int r1, r2/*, c1, c2*/;
  int nrow = nx * ny * nz;
  int ncol = nrow;
  int comm_size, comm_rank;
  pevsl_Csr csr_local;

  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  /* parcsr matrix allocation */
  /* when row_starts == NULL, will use the default 1D partitioning */
  /* when col_starts == NULL, will use the default 1D partitioning */
  pEVSL_ParcsrCreate(nrow, ncol, row_starts, col_starts, A, comm);
  
  r1 = A->first_row;
  r2 = A->first_row + A->nrow_local;
  /*
  c1 = A->first_col;
  c2 = A->first_col + A->ncol_local;
  */

  /* local rows of Lap */
  LocalLapGen(nx, ny, nz, r1, r2, &csr_local);
  
  /* setup parcsr matrix: create parcsr internal data and 
   * communication handle for matvec */
  pEVSL_ParcsrSetup(&csr_local, A);

  /* safe to destroy csr */
  pEVSL_FreeCsr(&csr_local);

  return 0;
}


/* @brief create a parallel csr matrix of 3-D Laplacians
 * For the same row and col starts [easier interface]
 */ 
int ParcsrLaplace(pevsl_Parcsr *A, int nx, int ny, int nz, int *row_col_starts, MPI_Comm comm) {
  int err = ParcsrLaplace2(A, nx, ny, nz, row_col_starts, row_col_starts, comm);
  return (err);
}

/**-----------------------------------------------------------------------
 * @brief Exact eigenvalues of Laplacean in interval [a b]
 * @param nx  Number of points in x-direction
 * @param ny  Number of points in y-direction
 * @param nz  Number of points in z-direction
 * @param[out] m number of eigenvalues found 
 * @param[out] **vo pointer to array of eigenvalues found 
 *-----------------------------------------------------------------------**/
int ExactEigLap3(int nx, int ny, int nz, double a, double b, int *m, double **vo) {
  double thetax = 0.5 * PI / (nx + 1.0);
  double thetay = 0.5 * PI / (ny + 1.0);
  double thetaz = 0.5 * PI / (nz + 1.0);
  int i, j, k, l=0, n=nx*ny*nz;
  double *v;
  double tx, ty, tz, ev;
  PEVSL_MALLOC(v, n, double);
  for (i=1; i<=nx; i++) {
    tx = sin(i*thetax);
    for (j=1; j<=ny; j++) {
      ty = sin(j*thetay);
      for (k=1; k<=nz; k++) {
        tz = sin(k*thetaz);
        if (1 == nz) {
          tz = 0.0;
        }
        ev = 4*(tx*tx+ty*ty+tz*tz);
        if (ev >= a - DBL_EPSILON && ev <= b + DBL_EPSILON) {
          v[l++] = ev;
        }
      }
    }
  }
  PEVSL_REALLOC(v, l, double);
  sort_double(l, v, NULL);

  *m = l;
  *vo = v;
  return 0;
}


