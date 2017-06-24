#ifndef PEVSL_STRUCT_H
#define PEVSL_STRUCT_H

#include <mpi.h>
#include <stdlib.h>

/**
 * @brief sparse matrix format: the coordinate (COO) format, 0-based
 *
 * ir, jc, vv : triples for all nonzeros (of size nnz)
 */
typedef struct _pevsl_cooMat {
  int nrows, ncols, nnz, *ir, *jc;
  double *vv;
} pevsl_Coo;


/*! 
 * @brief sparse matrix format: the compressed sparse row (CSR) format, 0-based
 * 
 * 3-array variant: ia,ja,a, nnz == ia[nrows]
 */ 
typedef struct _pevsl_csrMat {
  int nrows, ncols, *ia, *ja;
  double  *a;
} pevsl_Csr;

/*!
 * @brief macro to obtain number of nonzeros of a CSR matrix 
 */
#define PEVSL_CSRNNZ(A) A->ia[A->nrows]

/*! 
 * @brief communication handle used in parallel matvec
 * 
 */ 
typedef struct _commHandle {
  int num_proc_send_to; /**< num of procs to send data to */
  int *proc_send_to;    /**< ranks of these procs */
  int *send_elmts_ids;  /**< local ids of all sending data */
  int *send_elmts_ptr;  /**< starting ptr of each piece of sending data, size of num_proc_send_to+1 */

  int num_proc_recv_from; /**< num of procs to receive data from */
  int *proc_recv_from;    /**< ranks of these procs */
  int *recv_elmts_ptr;    /**< starting ptr of each piece of data reiceived [from each proc], size of num_proc_recv_from+1 */

  double *send_buf;
  double *recv_buf;
  MPI_Request *send_requests;
  MPI_Status *send_status;
  MPI_Request *recv_requests;
  MPI_Status *recv_status;
} commHandle;


/*! 
 * @brief parallel CSR format
 * 
 */ 
typedef struct _pevsl_parcsr {
  /**< mpi communicator that this matrix resides on */
  MPI_Comm comm;
  //MPI_Comm comm_global;
  //MPI_Comm comm_group;

  /* handle communications involved in parCSR matvec */
  commHandle *comm_handle;

  int nrow_global;
  int ncol_global;
  int nrow_local;
  int ncol_local;

  /* np: num of processors in comm, e.g., a group */
  /* array of length np+1, row_starts[i]: global index of the first row on proc i, row_starts[np] = nrow_global*/
  int *row_starts;
  /* array of length np+1, col_starts[i]: global index of the first column of the diag part on proc i, col_starts[np] = ncol_global */
  int *col_starts;

  /* my_first_row = row_starts[myrank],
   * my_first_col = col_starts[myrank]
   * my_row_range = [my_first_row : my_first_row + nrow_local-1]
   * my_col_range = [my_first_col : my_first_col + ncol_local-1]
   * 
   * For square matrix: row_range and col_range are often set to be the same
   * We reserve these for rectangular matrices (may not be useful EVSL though)
   *
   * NOTE: if row_starts or col_starts == NULL, it means that the default 1D partitioning is used
   * Using routine evsl_Part1d can easily decide all the info regarding the partitioning
   */

  int first_row;
  int first_col;

  /* diagonal    part of Ai    (local) : A(my_row_range, my_col_range) */
  pevsl_Csr *diag;

  /* off-diagonal part of Ai (external) : A(my_row_range, ~my_col_range) */
  pevsl_Csr *offd;

  /* global idx of the columns of offd, of size 'offd->ncols'
   * NOTE: columns are *local* indices. Use col_map_offd for mapping them back to global indices */
  int *col_map_offd;

} pevsl_Parcsr;

/*! 
 * @brief parallel vector
 * 
 */ 
typedef struct _pevsl_parvec {
  MPI_Comm comm;
  int n_global;
  int n_local;
  int n_first;
  double *data;
} pevsl_Parvec;


/*!
 * @brief timing and memory statistics of pEVSL
 *
 */
typedef struct _pevsl_Stat {
  /* timing [level-1 funcs] */
  double t_commgen;
  double t_eigbounds;
  double t_solver;
  /* timing [level-2 funcs] */
  double t_mvA;
  double t_mvB;
  double t_svB;
  size_t n_mvA;
  size_t n_mvB;
  size_t n_svB;
  /* memory */
  size_t alloced;
  size_t alloced_total;
  size_t alloced_max;
} pevsl_Stat;

/* global variable: pevsl_stats */
extern pevsl_Stat pevsl_stat;


/*!
 * @brief  parameters for polynomial filter 
 *
 * default values are set by set_pol_def
 */
typedef struct _pevsl_polparams {
  /** @name input to find_pol */
  /**@{*/
  int max_deg;        /**< max allowed degree */
  int min_deg ;       /**< min allowed degree */
  int damping;        /**< 0 = no damping, 1 = Jackson, 2 = Lanczos */
  double thresh_ext;  /**< threshold for accepting polynom. for end intervals */
  double thresh_int;  /**< threshold for interior intervals */
  /**@}*/

  /** @name output from find_pol */
  /**@{*/
  double *mu;         /**< coefficients. allocation done by set_pol */
  double cc;          /**< center of interval - used by chebAv */
  double dd;          /**< half-width of interval - used by chebAv */
  double gam;         /**< center of delta function used */
  double bar;         /**< p(theta)>=bar indicates a wanted Ritz value */
  /**@}*/

  /** @name both input to and output from find_pol */
  /**@{*/
  int deg ;           /**< if deg == 0 before calling find_deg then
                        the polynomial degree is  computed
                        internally. Otherwise it is of degree deg.
                        [and  thresh_ext and thresh_int are not used]
                        default value=0, set by call to set_pol_def */
  /**@}*/
} pevsl_Polparams;


/**
 * @brief linear solver function prototype: [complex version]
 * which is used for solving system with A-SIGMA B 
 * n  is the size  of the system,  br, bz are  the right-hand
 * side (real and  imaginary parts of complex vector),  xr, xz will
 * be the  solution (complex vector),  and "data" contains  all the
 * data  needed  by  the  solver. 
 */
typedef void (*SolFuncC)(pevsl_Parvec *br, pevsl_Parvec *bz, pevsl_Parvec *xr, pevsl_Parvec *xz, void *data);

/** 
 * @brief function prototype for applying the solve B x = b 
 * [the most general form]
 */
//typedef void (*SolFuncR)(pevsl_Parvec *b, pevsl_Parvec *x, void *data);
typedef void (*SolFuncR)(double *b, double *x, void *data);

/**
 * @brief matvec function prototype 
 */
//typedef void (*MVFunc)(pevsl_Parvec *x, pevsl_Parvec *y, void *data);
typedef void (*MVFunc)(double *x, double *y, void *data);

/*!
 * @brief user-provided Mat-Vec function and data for y = A * x or y = B * x
 *
 */
typedef struct _pevsl_Matvec {
  MVFunc func;         /**< function pointer */
  void *data;          /**< data */
} pevsl_Matvec;

/*!
 * @brief user-provided function and data for solving B x = b
 *
 */
typedef struct _pevsl_Bsol {
  SolFuncR func;       /**< function pointer */
  void *data;          /**< data */
} pevsl_Bsol;

/*!
 * @brief user-provided function for solving L^{T} x = b
 *
 */
typedef struct _pevsl_LtSol {
  SolFuncR func;       /**< function pointer */
  void *data;          /**< data */
} pevsl_Ltsol;

/*!
 * @brief data needed for Chebyshev iterations
 *
 */
typedef struct _BSolDataChebiter {
  /* eigenvalue bounds of B */
  double lb, ub;
  /* polynomial degree */
  int deg;
} BSolDataChebiter;


/*!
 * @brief wrapper of all global variables in pEVSL
 *
 */
typedef struct _pevsl_Data {
  /** We keep some useful information in this struct
   *  N is the global size of the problem, i.e, size of A (or B)
   *  n is the local size, number of rows owned by this MPI rank.
   *  If the matrix is assumed to be partitioned into blocks of consecutive rows,
   *  nfirst is the first row owned.
   *  NOTE that currently Parcsr matrix is always assumed to have such partitioning. 
   *  More general partitionings are NOT yet supported
   *  BUT, we leave the option to not have such consecutive row partitioning. 
   *  The users can implement their own Amv, Bmv, Bsol routines with arbitrary partitionings,
   *  and only set N and n but leave nfirst as PEVSL_NOT_DEFINED
   */
  int N;                    /**< global size of matrix A and B */
  int n;                    /**< local size of matrix A and B */
  int nfirst;               /**< the first local row and column */
  int ifGenEv;              /**< if it is a generalized eigenvalue problem */
  pevsl_Matvec *Amv;        /**< external matvec routine and the associated data for A */
  pevsl_Matvec *Bmv;        /**< external matvec routine and the associated data for B */
  pevsl_Bsol *Bsol;         /**< external function and data for B solve */
  pevsl_Ltsol *LTsol;       /**< external function and data for LT solve */
} pevsl_Data;


/* global variable: pevsl_data */
extern pevsl_Data pevsl_data;

#endif
