#include <float.h>
#include "pevsl.h"

int lapgen(int nx, int ny, int nz, int m1, int m2, pevsl_Csr *csr) {
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

int ParcsrLaplace(pevsl_Parcsr *A, int nx, int ny, int nz, int *row_col_starts_in, MPI_Comm comm) {
    int r1, r2;
    int nrow = nx * ny * nz;
    int *row_col_starts;
    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    // Assume that the same row and col partitionings are used
    row_col_starts = row_col_starts_in;
    // when row_col_starts == NULL, it will use the default 1D partitioning
    if (!row_col_starts_in) {
        // if not provided from the user, then use the default one
        pEVSL_Part1d(nrow, comm_size, &comm_rank, &r1, &r2, 1);
    } else {
        r1 = row_col_starts[comm_rank];
        r2 = row_col_starts[comm_rank+1];
    }

    /* matrix allocation */
    pEVSL_ParcsrCreate(nrow, nrow, row_col_starts, row_col_starts, A, comm);
    /* local rows */
    pevsl_Csr csr_local;
    lapgen(nx, ny, nz, r1, r2, &csr_local);
    /* setup parcsr matrix */
    pEVSL_ParcsrSetup(&csr_local, A);

    pEVSL_FreeCsr(&csr_local);

    return 0;
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


