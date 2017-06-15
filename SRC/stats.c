#include "pevsl_int.h"

void pEVSL_StatsPrint(FILE *fstats, MPI_Comm comm) {
  pevsl_Stat *stats = &pevsl_stat;
  /* time, max */
  double t_commgen, t_eigbounds, t_solver, t_mvA, t_mvB, t_svB;
  MPI_Reduce(&stats->t_commgen,   &t_commgen,   1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_eigbounds, &t_eigbounds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_solver,    &t_solver,    1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_mvA,       &t_mvA,       1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_mvB,       &t_mvB,       1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&stats->t_svB,       &t_svB,       1, MPI_DOUBLE, MPI_MAX, 0, comm);
  /* counts */
  unsigned long n_mvA, n_mvB, n_svB;
  n_mvA = stats->n_mvA;
  n_mvB = stats->n_mvB;
  n_svB = stats->n_svB;
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
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  /* group leader prints stats */
  if (rank == 0) {
    /* time */
    fprintf(fstats, "= = = = = = = = = = = = = = = = = = = Time & Memory Stats = = = = = = = = = = = = = = = = = = = =\n");
    fprintf(fstats, " Timing (sec):\n");
    fprintf(fstats, "   Create MPI communicators :  %f\n",  t_commgen);
    fprintf(fstats, "   Compute spectrum bounds  :  %f\n",  t_eigbounds);
    fprintf(fstats, "   Apply solver             :  %f\n",  t_solver);
    fprintf(fstats, "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
    fprintf(fstats, "   Matvec matrix A          :  %f (%8ld, avg %f)\n",  t_mvA, n_mvA, t_mvA / n_mvA);
    fprintf(fstats, "   Matvec matrix B          :  %f (%8ld, avg %f)\n",  t_mvB, n_mvB, t_mvB / n_mvB);
    fprintf(fstats, "   Solve with B             :  %f (%8ld, avg %f)\n",  t_svB, n_svB, t_svB / n_svB);
    /* memory */
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
    fprintf(fstats, "= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n");
  }
}

void pEVSL_StatsReset() {
  memset(&pevsl_stat, 0, sizeof(pevsl_Stat));
}
