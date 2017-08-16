#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "pevsl.h"
#include "common.h"
#include "pevsl_direct.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define OUTPUT_LEN 32768

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
  int n, nx, ny, nz, i, j, nslices, npts, nvec, Mdeg, nev, 
      ngroups, mlan, ev_int, sl, flg, ierr, np, rank;
  /* find the eigenvalues of A in the interval [a,b] */
  double a, b, lmax, lmin, ecount, tol, *sli, *mu, tm;
  double xintv[4];
  char *msg = NULL;
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
  /*-------------------- matrices A, B: parallel csr format */    
  pevsl_Parcsr A, B;
  /*-------------------- instance of pEVSL */
  pevsl_Data *pevsl;
  /*-------------------- Bsol */
  void *Bsol;
  /*-------------------- default values */
  nx = 8;
  ny = 8;
  nz = 8;
  a  = 1.5;
  b  = 2.5;
  nslices = 4;
  ngroups = 3;
  /*-----------------------------------------------------------------------
   *-------------------- reset some default values from command line  
   *                     user input from command line */
  flg = findarg("help", NA, NULL, argc, argv);
  if (flg && !rank) {
    printf("Usage: ./test -nx [int] -ny [int] -nz [int] -nslices [int] -ngroups [int] -a [double] -b [double]\n");
    return 0;
  }
  flg = findarg("nx", INT, &nx, argc, argv);
  flg = findarg("ny", INT, &ny, argc, argv);
  flg = findarg("nz", INT, &nz, argc, argv);
  flg = findarg("a", DOUBLE, &a, argc, argv);
  flg = findarg("b", DOUBLE, &b, argc, argv);
  flg = findarg("nslices", INT, &nslices, argc, argv);
  flg = findarg("ngroups", INT, &ngroups, argc, argv);
  /*-------------------- matrix size */
  n = nx * ny * nz;
  /*-------------------- stopping tol */
  tol = 1e-8;
  /*-------------------- Create communicators for groups, group leaders */
  CommInfoCreate(&comm, MPI_COMM_WORLD, ngroups);
  /*-------------------- Group leader (group_rank == 0) creates output file */
  if (comm.group_rank == 0) {
    /* output on screen */
    PEVSL_CALLOC(msg, OUTPUT_LEN, char);
    char fname[1024];
    sprintf(fname, "OUT/LapPLanN_G%d.out", comm.group_id);
    if (!(fstats = fopen(fname,"w"))) {
      PEVSL_ABORT(MPI_COMM_WORLD, -2, " failed in opening output file in OUT/\n");
    }
  }
  /*-------------------- output the problem settings */
  if (!comm.group_rank) {
    fprintf(fstats, "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    fprintf(fstats, "Laplacian A : %d x %d x %d, n = %d\n", nx, ny, nz, n);
    fprintf(fstats, "Laplacian B : %d x %d, n = %d\n", nx*ny*nz, 1, n);
    fprintf(fstats, "Interval: [%20.15f, %20.15f]  -- %d slices \n", a, b, nslices);
  }
  /*-------------------- generate 1D/3D Laplacian Parcsr matrices */
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //            Create Parcsr (Parallel CSR) matrices A and B
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // each proc group has a separate copy of parallel CSR A
  // the 5th argument is the row and col partitionings of A, 
  // i.e. row/col ranges for each proc, if NULL, trivial partitioning is used
  // [Important]: the last arg is the MPI_Comm that this matrix will reside on
  // so A is defined on each group
  ParcsrLaplace(&A, nx, ny, nz, NULL, comm.comm_group);
  ParcsrLaplace(&B, nx*ny*nz, 1, 1, NULL, comm.comm_group);
  /*-------------------- use MUMPS as the solver for B */
  SetupBSolDirect(&B, &Bsol);
  /*-------------------- start pEVSL
   * Create an instance of pEVSL on the GROUP communicator */
  pEVSL_Start(comm.comm_group, &pevsl);
  /*-------------------- set the left-hand side matrix A */
  pEVSL_SetAParcsr(pevsl, &A);
  /*-------------------- set the left-hand side matrix A */
  pEVSL_SetBParcsr(pevsl, &B);
  /*-------------------- set the solver for B and L^{T} */
  pEVSL_SetBSol(pevsl, BSolDirect, Bsol);
  
  //pEVSL_SetLTSol(pevsl, LTSolDirect, Bsol);
  
  void *Bpol;
  pEVSL_SetupLSPolSqrt(50, 1e-6, 2.0, 6.0, &B, &Bpol);
  pEVSL_SetLTSol(pevsl, pEVSL_LSPolSol, Bpol);  

  /*-------------------- for generalized eigenvalue problem */
  pEVSL_SetGenEig(pevsl);
  /*-------------------- step 0: get eigenvalue bounds */
  /*-------------------- random initial guess */
  pEVSL_ParvecCreate(A.ncol_global, A.ncol_local, A.first_col, comm.comm_group, &vinit);
  pEVSL_ParvecRand(&vinit);
  /*-------------------- bounds by TR-Lanczos */
  ierr = pEVSL_LanTrbounds(pevsl, 50, 200, 1e-8, &vinit, 1, &lmin, &lmax, fstats);
  if (!comm.group_rank) {
    fprintf(fstats, "Step 0: Eigenvalue bound s for B^{-1}*A: [%.15e, %.15e]\n", lmin, lmax);
  }
  /*-------------------- interval and eig bounds */
  xintv[0] = a;
  xintv[1] = b;
  xintv[2] = lmin;
  xintv[3] = lmax;
  /*-------------------- call kpmdos to get the DOS for dividing the spectrum*/
  /*-------------------- define kpmdos parameters */
  nvec = 60;
  Mdeg = 50;
  mu = (double *) malloc((Mdeg+1)*sizeof(double));
  /*-------------------- call KPM DOS */
#if 0
  tm = pEVSL_Wtime();
  ierr = pEVSL_Kpmdos(pevsl, Mdeg, 1, nvec, xintv, comm.ngroups, comm.group_id,
                      comm.comm_group_leader, mu, &ecount);
  tm = pEVSL_Wtime() - tm;
  fprintf(stdout, " KPMDOS: estimated eig count in interval: %f \n", ecount);
  if (ierr) {
    printf("kpmdos error %d\n", ierr);
    return 1;
  }
  if (comm.group_rank == 0) {
    fprintf(fstats, " Time to build DOS (kpmdos) was : %10.2f  \n", tm);
    fprintf(fstats, " estimated eig count in interval: %.15e \n", ecount);
  }
  /*-------------------- call Spslicer to slice the spectrum */
  npts = 10 * ecount;
  sli = (double *) malloc((nslices+1)*sizeof(double));
  if (comm.group_rank == 0) {
    fprintf(fstats, " DOS parameters: Mdeg = %d, nvec = %d, npnts = %d\n", Mdeg, nvec, npts);
  } 
  ierr = pEVSL_SpslicerKpm(sli, mu, Mdeg, xintv, nslices,  npts);
  /*-------------------- slicing done */
  if (ierr) {
    printf("spslicer error %d\n", ierr);
    return 1;
  }
#else
  Mdeg = 50;
  npts = 200;
  double *xdos = (double *)calloc(npts, sizeof(double));
  double *ydos = (double *)calloc(npts, sizeof(double));
  tm = pEVSL_Wtime();
  pEVSL_LanDosG(pevsl, nvec, Mdeg, npts, xdos, ydos, &ecount, xintv,
                comm.ngroups, comm.group_id, comm.comm_group_leader);
  tm = pEVSL_Wtime() - tm;
  fprintf(stdout, " LANDOS: estimated eig count in interval: %f \n", ecount);
  if (comm.group_rank == 0) {
    fprintf(fstats, " Time to build DOS (Lanczos dos) was : %10.2f  \n", tm);
    fprintf(fstats, " estimated eig count in interval: %.15e \n", ecount);
  }
  /*-------------------- call Spslicer to slice the spectrum */
  sli = (double *) malloc((nslices+1)*sizeof(double));
  pEVSL_SpslicerLan(xdos, ydos, nslices, npts, sli);
  PEVSL_FREE(xdos);
  PEVSL_FREE(ydos);
#endif

  if (comm.group_rank == 0) {
    fprintf(fstats, "====================  SLICES FOUND  ====================\n");
    for (j=0; j<nslices;j++) {
      fprintf(fstats, " %2d: [% .15e , % .15e]\n", j+1, sli[j],sli[j+1]);
    }
    fprintf(fstats, "========================================================\n");
  }
  /*--------------------- print stats */
  pEVSL_StatsPrint(pevsl, fstats);
  /*-------------------- # eigs per slice */
  ev_int = (int) (1 + ecount / ((double) nslices));
  
  /*------------------- For each slice call ChebLanr, each group pick one slice */
  for (sl=comm.group_id; sl<nslices; sl+=comm.ngroups) {
    int nev2, *ind;
    double *lam, *res, ai, bi;
    pevsl_Parvecs *Y;
    /*-------------------- zero out stats */
    pEVSL_StatsReset(pevsl);
    /*-------------------- */
    ai = sli[sl];
    bi = sli[sl+1];
    /*-------------------- approximate number of eigenvalues wanted */
    nev = ev_int+2;
    /*-------------------- Dimension of Krylov subspace */
    mlan = max(5*nev, 300);
    mlan = min(mlan, n);
    /*-------------------- Interval */
    xintv[0] = ai;    xintv[1] = bi;  
    xintv[2] = lmin;  xintv[3] = lmax;
    //-------------------- set up default parameters for pol.      
    pEVSL_SetPolDef(&pol);
    //-------------------- this is to show how you can reset some of the
    //                     parameters to determine the filter polynomial
    pol.damping = 2;
    //-------------------- use a stricter requirement for polynomial
    pol.thresh_int = 0.6;
    pol.thresh_ext = 0.15;
    pol.max_deg  = 3000;
    // pol.deg = 20 //<< this will force this exact degree . not recommended
    //                   it is better to change the values of the thresholds
    //                   pol.thresh_ext and plot.thresh_int
    //-------------------- Now determine polymomial to use
    pEVSL_FindPol(xintv, &pol);
    
    if (comm.group_rank == 0) {
      fprintf(fstats, "\n\n");
      fprintf(fstats, " ======================================================\n");
      fprintf(fstats, " subinterval %3d: [%.4e , %.4e]\n", sl, ai, bi);
      fprintf(fstats, " ======================================================\n");
      fprintf(fstats, " polynomial [type %d] deg %d, bar %e gam %e\n", pol.type, pol.deg, pol.bar, pol.gam);
    }
    //-------------------- then call ChenLanNr
    ierr = pEVSL_ChebLanNr(pevsl, xintv, mlan, tol, &vinit, &pol, &nev2, &lam, &Y, &res, fstats);
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
      fprintf(fstats, " [Group %d]: number of eigenvalues found: %d\n", 
              comm.group_id, nev2);
      
      sprintf(msg+strlen(msg), " ======================================================\n");
      sprintf(msg+strlen(msg), " subinterval %3d: [%.4e , %.4e]\n", sl, ai, bi);
      sprintf(msg+strlen(msg), " [Group %d]: number of eigenvalues found: %d\n",
              comm.group_id, nev2);

      /* print eigenvalues */
      fprintf(fstats, "     Eigenvalues in [a, b]\n");
      fprintf(fstats, "     Computed [%d]       ||Res||\n", nev2);
      for (i=0; i<nev2; i++) {
        fprintf(fstats, "% .15e  %.1e\n", lam[i], res[ind[i]]);
        if (i>50) {
          fprintf(fstats,"                        -- More not shown --\n");
          break;
        }
      }
    }
    /*-------------------- free within this slice */
    if (lam) { free(lam); }
    if (Y) {
      pEVSL_ParvecsFree(Y);
      free(Y);
    }
    if (res) { free(res); }
    pEVSL_FreePol(&pol);
    free(ind);
    /*--------------------- print stats */
    pEVSL_StatsPrint(pevsl, fstats);
  } /* for (sl=0 */
    
  /* group leaders print on screen in orders */
  if (comm.group_rank == 0) {
    PEVSL_SEQ_BEGIN(comm.comm_group_leader);
    fprintf(stdout, "%s", msg);
    PEVSL_SEQ_END(comm.comm_group_leader);
  }

  /*--------------------- done */
  if (fstats) {
    fclose(fstats);
  }
  free(sli);
  free(mu);
  
  FreeBSolDirectData(Bsol);
  pEVSL_ParcsrFree(&A);
  pEVSL_ParcsrFree(&B);
  pEVSL_ParvecFree(&vinit);
  CommInfoFree(&comm);

  pEVSL_Finish(pevsl);
  MPI_Finalize();

  return 0;
}

