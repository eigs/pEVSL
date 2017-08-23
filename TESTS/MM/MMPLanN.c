#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "pevsl.h"
#include "common.h"
#if USE_MKL
#include "mkl.h"
#endif

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define OUTPUT_LEN 32768

int main(int argc, char *argv[]) {
/*-----------------------------------------------------------------
  This example tests Non-restart Lanczos with polynomial filtering
  with general matrices
-------------------------------------------------------------------*/
  int n, mat, i, j, npts, nslices, nvec, Mdeg, nev, 
      ngroups, mlan, ev_int, sl, flg, ierr, np, rank, dostype;
  /* find the eigenvalues of A in the interval [a,b] */
  double lmax, lmin, ecount, tol, *sli, *mu;
  char matfile[4096] = "matfile";
  double xintv[4];
  double tm;
  io_t io;
  dostype = 0; /* 0: KPM, 1: Lanczos */ 

#if USE_MKL
  mkl_set_dynamic(1);
  mkl_set_num_threads(12);
#endif
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
  /*-------------------- matrix A: parallel csr format */    
  pevsl_Parcsr A;
  /*-------------------- instance of pEVSL */
  pevsl_Data *pevsl;
  /*-------------------- default values */
  ngroups = 3;
  /*-----------------------------------------------------------------------
   *-------------------- reset some default values from command line  
   *                     user input from command line */
  flg = findarg("help", NA, NULL, argc, argv);
  if (flg && !rank) {
    printf("Usage: ./test -matfile [filename] -ngroups [int] -dostype [0/1] \n");
    return 0;
  }
  flg = findarg("matfile", STR, matfile, argc, argv);
  flg = findarg("ngroups", INT, &ngroups, argc, argv);
  flg = findarg("dostype", INT, &dostype, argc, argv);
  /*-------------------- stopping tol */
  tol = 1e-8;
  /*-------------------- Partition MPI_COMM_WORLD into ngroups subcomm's,
   * create a communicator for each group, and one for group leaders
   * saved in comm */
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
  /*------------------ file "matfile" contains paths to matrices */
  /* job = 0, open matfile and get the number of matrices */
  ierr = GetMatfileInfo(&io, 0, matfile); PEVSL_CHKERR(ierr);
  /*------------------ matrix loop */
  for (mat = 0; mat < io.ntests; mat++) {
    /* job = 1, load a problem line */
    ierr = GetMatfileInfo(&io, 1, NULL); PEVSL_CHKERR(ierr);
    nslices = io.nslices;
    /* load parcsr matrix A from an MM file */
    ParcsrReadMM(&A, io.Fname1, io.Fmt==MM1, NULL, comm.comm_group);
    if (comm.global_rank == 0) {
      fprintf(stdout, " Problem %3d: A: %s(%s), Format: %s, intv [%e, %e], nslices %d \n", 
              mat, io.MatNam1, io.Fname1, io.FmtStr, io.a, io.b, io.nslices);
    }
    if (comm.group_rank == 0) {
      fprintf(fstats, " Problem %3d: A: %s(%s), Format: %s, intv [%e, %e], nslices %d \n", 
              mat, io.MatNam1, io.Fname1, io.FmtStr, io.a, io.b, io.nslices);
    }
    /*-------------------- start pEVSL 
     * Create an instance of pEVSL on the GROUP communicator */
    pEVSL_Start(comm.comm_group, &pevsl);
    /*-------------------- set matrix A */
    pEVSL_SetAParcsr(pevsl, &A);
    n = A.nrow_global;
    /*-------------------- step 0: get eigenvalue bounds */
    /*-------------------- random initial guess */
    pEVSL_ParvecCreate(A.ncol_global, A.ncol_local, A.first_col, comm.comm_group, &vinit);
    pEVSL_ParvecRand(&vinit);
    /*-------------------- bounds by TR-Lanczos */
    ierr = pEVSL_LanTrbounds(pevsl, 50, 200, 1e-8, &vinit, 1, &lmin, &lmax, fstats);
    if (!comm.group_rank) {
      fprintf(fstats, "Step 0: Eigenvalue bound s for A: [%.15e, %.15e]\n", lmin, lmax);
    }
    /*-------------------- interval and eig bounds */
    xintv[0] = io.a;
    xintv[1] = io.b;
    xintv[2] = lmin;
    xintv[3] = lmax;
    /*-------------------- DOS */
    nvec = 60;
    Mdeg = 50;
    mu = (double *) malloc((Mdeg+1)*sizeof(double));
    if (dostype == 0) {
      /*-------------------- call KPM DOS */
      tm = pEVSL_Wtime();
      ierr = pEVSL_Kpmdos(pevsl, Mdeg, 1, nvec, xintv, comm.ngroups, comm.group_id,
                          comm.comm_group_leader, mu, &ecount);
      tm = pEVSL_Wtime() - tm;
      if (comm.global_rank == 0) {
        fprintf(stdout, " KPMDOS: estimated eig count in interval: %f \n", ecount);
        fprintf(fstats, " KPMDOS: estimated eig count in interval: %f \n", ecount);
      }
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
        fprintf(fstats, " DOS parameters: Mdeg = %d, nvec = %d, npnts = %d\n",
                Mdeg, nvec, npts);
      }
      ierr = pEVSL_SpslicerKpm(sli, mu, Mdeg, xintv, nslices,  npts);
      /*-------------------- slicing done */
      if (ierr) {
        printf("spslicer error %d\n", ierr);
        return 1;
      }
    } else {
      /*-------------------- call Lanczos DOS */
      int msteps = 40;
      npts = 200;
      double *xdos = (double *)calloc(npts, sizeof(double));
      double *ydos = (double *)calloc(npts, sizeof(double));
      tm = pEVSL_Wtime();
      pEVSL_LanDosG(pevsl, nvec, msteps, npts, xdos, ydos, &ecount, xintv,
                    comm.ngroups, comm.group_id, comm.comm_group_leader);
      tm = pEVSL_Wtime() - tm;
      if (comm.global_rank == 0) {
        fprintf(stdout, " LanDOS: estimated eig count in interval: %f \n", ecount);
        fprintf(fstats, " LanDOS: estimated eig count in interval: %f \n", ecount);
      }
      if (comm.group_rank == 0) {
        fprintf(fstats, " Time to build DOS (Lanczos dos) was : %10.2f  \n", tm);
        fprintf(fstats, " estimated eig count in interval: %.15e \n", ecount);
      }
      /*-------------------- call Spslicer to slice the spectrum */
      sli = (double *) malloc((nslices+1)*sizeof(double));
      pEVSL_SpslicerLan(xdos, ydos, nslices, npts, sli);
      PEVSL_FREE(xdos);
      PEVSL_FREE(ydos);
    }
     
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
        fprintf(fstats, " ======================================================\n");
        fprintf(fstats, " subinterval %3d: [%.4e , %.4e]\n", sl, ai, bi);
        fprintf(fstats, " ======================================================\n");
        fprintf(fstats, " polynomial [type %d] deg %d, bar %e gam %e\n",
                pol.type, pol.deg, pol.bar, pol.gam);
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
    } /* slice loop, for (sl=0 */

    /* group leaders print on screen in orders */
    if (comm.group_rank == 0) {
      PEVSL_SEQ_BEGIN(comm.comm_group_leader);
      fprintf(stdout, "%s", msg);
      PEVSL_SEQ_END(comm.comm_group_leader);
    }

    free(sli);
    free(mu);
    pEVSL_ParcsrFree(&A);
    pEVSL_ParvecFree(&vinit);
    pEVSL_Finish(pevsl);
    if (comm.group_rank == 0) {
      fprintf(fstats, "\n= = = = = = = = = pEVSL Done = = = = = = = = =\n");
    }
    /*---------------- clear on screen output */
    if (msg) { msg[0] = 0; }
  } /* matrix loop */

  /* job = 2, close matfile */
  ierr = GetMatfileInfo(&io, 2, NULL); PEVSL_CHKERR(ierr);

  /*--------------------- done */
  if (fstats) {
    fclose(fstats);
  }
  CommInfoFree(&comm);
  if (msg) {
    free(msg);
  }
  MPI_Finalize();

  return 0;
}

