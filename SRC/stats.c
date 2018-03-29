#include "pevsl_int.h"

/**
 * @file stats.c
 * @brief Used to track various statistics (time taken by various
 * operations).
 */
/** 
 * @brief Prints out stats
 * @param[in] pevsl pevsl data struct
 * @param[in] fstats FILE to print to
 */
void pEVSL_StatsPrint(pevsl_Data *pevsl, FILE *fstats) {
  pevsl_Stat *stats = pevsl->stats;
  MPI_Comm comm = pevsl->comm;
  /* time, max */
  double t_setBsv, t_setASigBsv, t_eigbounds, t_dos, t_iter, t_mvA, t_mvB, t_svB, 
         t_svLT, t_svASigB, t_reorth, t_eig, t_blas, t_ritz, t_polAv, t_ratAv, t_sth;

  MPI_Reduce(&stats->t_setBsv,     &t_setBsv,     1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_setASigBsv, &t_setASigBsv, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_eigbounds,  &t_eigbounds,  1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_dos,        &t_dos,        1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_iter,       &t_iter,       1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_mvA,        &t_mvA,        1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_mvB,        &t_mvB,        1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_svB,        &t_svB,        1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_svLT,       &t_svLT,       1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_svASigB,    &t_svASigB,    1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_reorth,     &t_reorth,     1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_eig,        &t_eig,        1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_blas,       &t_blas,       1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_ritz,       &t_ritz,       1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_polAv,      &t_polAv,      1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_ratAv,      &t_ratAv,      1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_sth,        &t_sth,        1, MPI_DOUBLE, MPI_MAX, 0, comm);
  /* counts */
  unsigned long n_mvA, n_mvB, n_svB, n_svLT, n_svASigB, n_polAv, n_ratAv;
  n_mvA = stats->n_mvA;
  n_mvB = stats->n_mvB;
  n_svB = stats->n_svB;
  n_svLT = stats->n_svLT;
  n_svASigB = stats->n_svASigB;
  n_polAv = stats->n_polAv;
  n_ratAv = stats->n_ratAv;
  /* memory, sum */
  unsigned long alloced, alloced_total, alloced_max, myalloced, myalloced_total, myalloced_max;
  myalloced = stats->alloced;
  myalloced_total = stats->alloced_total;
  myalloced_max = stats->alloced_max;
  MPI_Reduce(&myalloced,       &alloced,       1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
  MPI_Reduce(&myalloced_total, &alloced_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
  MPI_Reduce(&myalloced_max,   &alloced_max,   1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
  /* rank 0 prints */
  int rank;
  MPI_Comm_rank(comm, &rank);
  /* group leader prints stats */
  if (rank == 0) {
    /* time */
    fprintf(fstats, "= = = = = = = = = = = = = = = = = = = Time & Memory Stats = = = = = = = = = = = = = = = = = = = =\n");

    fprintf(fstats, " Timing (sec):\n");
    if (t_setBsv)     { fprintf(fstats, "   Setup Solver for B        :  %f\n",  t_setBsv); }
    if (t_setASigBsv) { fprintf(fstats, "   Setup Solver for A-SIG*B  :  %f\n",  t_setASigBsv); }
    if (t_setASigBsv) { fprintf(fstats, "   Compute Eigenvalue bounds :  %f\n",  t_eigbounds); }
    if (t_dos)        { fprintf(fstats, "   Compute DOS               :  %f\n",  t_dos); }
    if (t_iter)       { fprintf(fstats, "   Iteration time (tot)      :  %f\n",  t_iter); }

    fprintf(fstats, "   - - - - - - - - - - - - - - - - -\n");

    if (n_polAv)   { fprintf(fstats, "   Pol(A)*v                  :  %f (%8ld, avg %f)\n",  t_polAv, n_polAv, t_polAv / n_polAv); }
    if (n_ratAv)   { fprintf(fstats, "   Rat(A)*v                  :  %f (%8ld, avg %f)\n",  t_ratAv, n_ratAv, t_ratAv / n_ratAv); }
    if (n_mvA)     { fprintf(fstats, "   Matvec matrix A           :  %f (%8ld, avg %f)\n",  t_mvA, n_mvA, t_mvA / n_mvA); }
    if (n_mvB)     { fprintf(fstats, "   Matvec matrix B           :  %f (%8ld, avg %f)\n",  t_mvB, n_mvB, t_mvB / n_mvB); }
    if (n_svB)     { fprintf(fstats, "   Solve with B              :  %f (%8ld, avg %f)\n",  t_svB, n_svB, t_svB / n_svB); }
    if (n_svLT)    { fprintf(fstats, "   Solve with LT             :  %f (%8ld, avg %f)\n",  t_svLT, n_svLT, t_svLT / n_svLT); }
    if (n_svASigB) { fprintf(fstats, "   Solve with A-SIGMA*B      :  %f (%8ld, avg %f)\n",  t_svASigB, n_svASigB, t_svASigB / n_svASigB); }
    if (t_reorth)  { fprintf(fstats, "   Reorthogonalization       :  %f\n", t_reorth); }
    if (t_eig)     { fprintf(fstats, "   LAPACK eig                :  %f\n", t_eig); }
    if (t_blas)    { fprintf(fstats, "   Other BLAS                :  %f\n", t_blas); }
    if (t_ritz)    { fprintf(fstats, "   Compute Ritz vectors      :  %f\n", t_ritz); }
    if (t_sth)     { fprintf(fstats, "   Other                     :  %f\n", t_sth); }

    /* memory */
    /*
       if (alloced_total > 1e9) {
       fprintf(fstats, " Memory (GB):\n");
       fprintf(fstats, "   Total %.2f,  Peak %.2f \n", alloced_total/1e9, alloced_max/1e9);
       } else if (alloced_total > 1e6) {
       fprintf(fstats, " Memory (MB):\n");
       fprintf(fstats, "   Total %.2f,  Peak %.2f \n", alloced_total/1e6, alloced_max/1e6);
       } else {
       fprintf(fstats, " Memory (KB):\n");
       fprintf(fstats, "   Total %.2f,  Peak %.2f \n", alloced_total/1e3, alloced_max/1e3);
       }
       if (alloced) {
       fprintf(fstats, "warning: unfreed memory %ld\n", alloced);
       }
       fflush(fstats);
       */
    fprintf(fstats, "= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
  }
}

/**
 * @brief Resets stats
 * @param[in, out] pevsl data struct
 */
void pEVSL_StatsReset(pevsl_Data *pevsl) {
  pevsl_Stat *stats = pevsl->stats;
  memset(stats, 0, sizeof(pevsl_Stat));
}
