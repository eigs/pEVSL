#ifndef TEST_COMMON_H
#define TEST_COMMON_H

/*! 
 * @brief informations on communicators
 * 
 */ 
typedef struct _CommInfo {
  MPI_Comm comm_global;       // global communicator (MPI_COMM_WORLD) 
  MPI_Comm comm_group;        // communicator within each group
  MPI_Comm comm_group_leader; // communicator with the rank-0's of all groups
  // global info
  int global_size;            // number of processors in comm_global
  int global_rank;            // rank in comm_global
  // group info
  int ngroups;                // number of processor/sub-communicator groups
  int group_id;               // group index
  int group_size;             // number of processors in comm_group
  int group_rank;             // rank in comm_group
} CommInfo;

#include "io.h"

int CommInfoCreate(CommInfo *comm, MPI_Comm comm_global, int ngroups);

void CommInfoFree(CommInfo *comm);

int ParcsrLaplace(pevsl_Parcsr *A, int nx, int ny, int nz, int *row_col_starts_in, MPI_Comm comm);

int ExactEigLap3(int nx, int ny, int nz, double a, double b, int *m, double **vo);

#endif
