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

int LocalLapGen(int nx, int ny, int nz, int m1, int m2, pevsl_Csr *csr);

int ParcsrLaplace(pevsl_Parcsr *A, int nx, int ny, int nz, int *row_col_starts, MPI_Comm comm);

int ParcsrLaplace2(pevsl_Parcsr *A, int nx, int ny, int nz, int *row_starts, int *col_starts, MPI_Comm comm);

int ExactEigLap3(int nx, int ny, int nz, double a, double b, int *m, double **vo);

int RandElems(int n, int m, int *elem);

void SpRandCsr(int nrow, int ncol, int rownnz, pevsl_Csr *csr);

#endif
