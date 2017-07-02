#include <stdio.h>
#include <mpi.h>
#include "pevsl.h"
#include "common.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

int main(int argc, char *argv[]) {
/*------------------------------------------------------------
  generates a laplacean matrix on an nx x ny x nz mesh 
  and computes all eigenvalues in a given interval [a  b]
  The default set values are
  nx = 41; ny = 53; nz = 1;
  a = 0.4; b = 0.8;
  nslices = 1 [one slice only] 
  other parameters 
  tol [tolerance for stopping - based on residual]
  Mdeg = pol. degree used for DOS
  nvec  = number of sample vectors used for DOS
  This uses:
  Non-restart Lanczos with polynomial filtering
------------------------------------------------------------*/
  int n, nx, ny, nz, i, j, npts, nslices, nvec, Mdeg, nev, 
      ngroups, mlan, ev_int, sl, flg, ierr, np, rank;
  /* find the eigenvalues of A in the interval [a,b] */
  double a, b, lmax, lmin, ecount, tol, *sli, *mu;
  double xintv[4];
  double *xdos, *ydos;
  /*-------------------- communicator struct, which contains all the communicators */
  CommInfo comm;
  pevsl_Parvec vinit;
  pevsl_Polparams pol;
  FILE *fstats = NULL;
  /*--------------------- Initialize MPI */
  int rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS) {
    printf ("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
  /*-------------------- size and rank */
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  /*-------------------- matrix A: parallel csr format */    
  pevsl_Parcsr A;
  /*-------------------- default values */
  nx = 16;
  ny = 16;
  nz = 16;
  a  = 0.4;
  b  = 0.5;
  nslices = 1;
  ngroups = 1;
  /*-----------------------------------------------------------------------
   *-------------------- reset some default values from command line  
   *                     user input from command line */
  flg = findarg("help", NA, NULL, argc, argv);
  if (flg && !rank) {
    printf("Usage: ./test -nx [int] -ny [int] -nz [int] -nslices [int] -a [double] -b [double]\n");
    return 0;
  }
  flg = findarg("nx", INT, &nx, argc, argv);
  flg = findarg("ny", INT, &ny, argc, argv);
  flg = findarg("nz", INT, &nz, argc, argv);
  flg = findarg("a", DOUBLE, &a, argc, argv);
  flg = findarg("b", DOUBLE, &b, argc, argv);
  flg = findarg("nslices", INT, &nslices, argc, argv);
  /*-------------------- eigenvalue bounds set by hand */
  lmin = 0.0;  
  lmax = nz == 1 ? 8.0 : 12.0;
  xintv[0] = a;
  xintv[1] = b;
  xintv[2] = lmin;
  xintv[3] = lmax;
  tol  = 1e-8;
  n = nx * ny * nz;
  /*-------------------- Partition MPI_COMM_WORLD into ngroups subcomm's,
   * create a communicator for each group, and one for group leaders
   * saved in comm */
  CommInfoCreate(&comm, MPI_COMM_WORLD, ngroups);
  /*-------------------- Group leader (group_rank == 0) creates output file */
  if (comm.group_rank == 0) {
    char fname[1024];
    sprintf(fname, "OUT/LapPLanN_G%d.out", comm.group_id);
    if (!(fstats = fopen(fname,"w"))) {
      printf(" failed in opening output file in OUT/\n");
      fstats = stdout;
    }
    fprintf(fstats, " nx %d ny %d nz %d, nslices %d, a = %e b = %e\n", 
            nx, ny, nz, nslices, a, b);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //            Create Parcsr (Parallel CSR) matrix A
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // each proc group has a separate copy of parallel CSR A
  // the 5th argument is the row and col partitionings of A, 
  // i.e. row/col ranges for each proc, if NULL, trivial partitioning is used
  // [Important]: the last arg is the MPI_Comm that this matrix will reside on
  // so A is defined on each group
  ParcsrLaplace(&A, nx, ny, nz, NULL, comm.comm_group);
  /*-------------------- start pEVSL */
  pEVSL_Start(argc, argv);
  /*-------------------- set matrix A */
  pEVSL_SetAParcsr(&A);
  /*-------------------- call DOS */
  mu = (double *) malloc((Mdeg+1)*sizeof(double));
  sli = (double *) malloc((nslices+1)*sizeof(double));
  /*------------------- trivial */
  linspace(a, b, nslices+1, sli);      
  /*------------------- Create parallel vector: random initial guess */
  pEVSL_ParvecCreate(A.ncol_global, A.ncol_local, A.first_col, comm.comm_group, &vinit);
  pEVSL_ParvecRand(&vinit);

  //double ll, mm;
  //pEVSL_LanTrbounds(50, 200, 1e-10, &vinit, 1, &ll, &mm, comm.comm_group, NULL);
  //printf("lmin = %.15e, lmax = %.15e\n", ll, mm);
  //exit(0);

  /*------------------- For each slice call ChebLanr */
  for (sl=comm.group_id; sl<nslices; sl+=comm.ngroups) {
    int nev2, *ind, nev_ex;
    double *lam, *res, *lam_ex, ai, bi;
    pevsl_Parvec *Y;
    /*-------------------- */
    ai = sli[sl];
    bi = sli[sl+1];
    mlan = min(500, n);
    xintv[0] = ai;  
    xintv[1] = bi;  
    xintv[2] = lmin;  
    xintv[3] = lmax;
    //-------------------- set up default parameters for pol.      
    pEVSL_SetPolDef(&pol);
    //-------------------- this is to show how you can reset some of the
    //                     parameters to determine the filter polynomial
    pol.damping = 1;
    //-------------------- use a stricter requirement for polynomial
    pol.thresh_int = 0.5;
    pol.thresh_ext = 0.15;
    pol.max_deg  = 500;
    //-------------------- Now determine polymomial to use
    pEVSL_FindPol(xintv, &pol);
    if (comm.group_rank == 0) {
      fprintf(fstats, " ======================================================\n");
      fprintf(fstats, " subinterval: [%.4e , %.4e]\n", ai, bi);
      fprintf(fstats, " ======================================================\n");
      fprintf(fstats, " polynomial deg %d, bar %e gam %e\n", pol.deg, pol.bar, pol.gam);
    }
    //-------------------- then call ChenLanNr    
    ierr = pEVSL_ChebLanNr(xintv, mlan, tol, &vinit, &pol, &nev2, &lam, &Y, &res, 
                           comm.comm_group, fstats);
    if (ierr) {
      printf("ChebLanNr error %d\n", ierr);
      return 1;
    }
    /*-------------------- output results */
    /* sort the eigenvals: ascending order
     * ind: keep the orginal indices */
    ind = (int *) malloc(nev2*sizeof(int));
    sort_double(nev2, lam, ind);
    /* group leader checks the eigenvalues and print */
    if (comm.group_rank == 0) {
      /* compute exact eigenvalues */
      ExactEigLap3(nx, ny, nz, ai, bi, &nev_ex, &lam_ex);
      fprintf(fstats, " [Group %d]: number of eigenvalues: %d, found: %d\n", 
              comm.group_id, nev_ex, nev2);
      if (fstats != stdout) {
        fprintf(stdout, " [Group %d]: number of eigenvalues: %d, found: %d\n",
                comm.group_id, nev_ex, nev2);
      } 
      /* print eigenvalues */
      fprintf(fstats, "                                   Eigenvalues in [a, b]\n");
      fprintf(fstats, "     Computed [%d]       ||Res||              Exact [%d]", nev2, nev_ex);
      if (nev2 == nev_ex) {
        fprintf(fstats, "                 Err");
      }
      fprintf(fstats, "\n");
      for (i=0; i<max(nev2, nev_ex); i++) {
        if (i < nev2) {
          fprintf(fstats, "% .15e  %.1e", lam[i], res[ind[i]]);
        } else {
          fprintf(fstats, "                               ");
        }
        if (i < nev_ex) { 
          fprintf(fstats, "        % .15e", lam_ex[i]);
        }
        if (nev2 == nev_ex) {
          fprintf(fstats, "        % .1e", lam[i]-lam_ex[i]);
        }
        fprintf(fstats,"\n");
        if (i>50) {
          fprintf(fstats,"                        -- More not shown --\n");
          break;
        } 
      }
      free(lam_ex);
    }
    /*-------------------- free within this slice */
    if (lam) free(lam);
    if (res) free(res);
    pEVSL_FreePol(&pol);
    free(ind);
    for (i=0; i<nev2; i++) {
      pEVSL_ParvecFree(&Y[i]);
    }
    free(Y);
  } /* for (sl=0 */
  
  if (fstats) fclose(fstats);
  free(sli);
  free(mu);
  pEVSL_ParcsrFree(&A);
  pEVSL_ParvecFree(&vinit);

  CommInfoFree(&comm);

  pEVSL_Finish();

  MPI_Finalize();

  return 0;
}

