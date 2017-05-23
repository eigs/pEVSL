#ifndef PEVSL_DEF_H
#define PEVSL_DEF_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#define PEVSL_NOT_IMPLEMENT 11
#define PEVSL_MAX_LINE 2048

#define PEVSL_MAX(a, b) ((a) > (b) ? (a) : (b))
#define PEVSL_MIN(a, b) ((a) < (b) ? (a) : (b))

#define orthTol 1e-14

#define PI M_PI

#define PEVSL_CHKERR(ierr) assert(!(ierr))

#define PEVSL_ABORT(comm, errcode, msg) {\
    int pid; \
    MPI_Comm_rank(comm, &pid); \
    printf("PEVSL error (processor %d): %s \n", pid, msg);  \
    MPI_Abort(comm, errcode); \
}


/* memory management, alloc and free */

#define PEVSL_MALLOC(base, nmem, type) {\
  (base) = (type *)malloc((nmem)*sizeof(type)); \
  PEVSL_CHKERR((base) == NULL); \
}
#define PEVSL_CALLOC(base, nmem, type) {\
  (base) = (type *)calloc((nmem), sizeof(type)); \
  PEVSL_CHKERR((base) == NULL); \
}
#define PEVSL_REALLOC(base, nmem, type) {\
  (base) = (type *)realloc((base), (nmem)*sizeof(type)); \
  PEVSL_CHKERR((base) == NULL && nmem > 0); \
}

#define PEVSL_FREE(base) {\
    free(base);\
}


#define PEVSL_SEQ_BEGIN(MPI_COMM, __RANK) {\
    int __comm_size, __comm_rank, __i;\
    MPI_Barrier(MPI_COMM);\
    MPI_Comm_size(MPI_COMM, &__comm_size);\
    MPI_Comm_rank(MPI_COMM, &__comm_rank);\
    for (__i=0; __i<__comm_size; __i++) {\
        if (__i == __comm_rank && (__RANK < 0 || __comm_rank == __RANK)) {\

#define PEVSL_SEQ_END(MPI_COMM) \
        }\
        MPI_Barrier(MPI_COMM);\
    }\
}

/*!
  \def max(x,y)
  Computes the maximum of \a x and \a y.
*/
#define PEVSL_MAX(a, b) ((a) > (b) ? (a) : (b))

/*!
  \def min(x,y)
  Computes the minimum of \a x and \a y.
*/
#define PEVSL_MIN(a, b) ((a) < (b) ? (a) : (b))

/*! Fortran interface naming convention
 */
#define PEVSL_FORT(name) name ## _f90_

/*! max number of Gram–Schmidt process in orthogonalization
 */
#define NGS_MAX 2

#endif

