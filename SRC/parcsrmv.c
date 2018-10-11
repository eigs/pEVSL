#include "pevsl_int.h"

/**
 * @file parcsrmv.c
 * @brief Parallel csr matrix vector products
 */

/**
 * @brief Beginning function for matvecs
 * @param[in] A csr matrix
 * @param[in] x vector
 */
void pEVSL_ParcsrMatvecCommBegin(pevsl_Parcsr *A, double *x) {
    int i;
    int np_send = A->comm_handle->num_proc_send_to;
    int np_recv = A->comm_handle->num_proc_recv_from;

    /* post non-blocking recv */
    for (i=0; i<np_recv; i++) {
        int p = A->comm_handle->proc_recv_from[i];
        int j1 = A->comm_handle->recv_elmts_ptr[i];
        int j2 = A->comm_handle->recv_elmts_ptr[i+1];
        PEVSL_CHKERR(j2-j1 <= 0);
        MPI_Irecv(A->comm_handle->recv_buf+j1, j2-j1, MPI_DOUBLE, p, 0, \
                  A->comm, A->comm_handle->send_recv_requests+np_send+i);
    }

    /* copy elements to send into send buffer */
    int nelmts_send = A->comm_handle->send_elmts_ptr[np_send];
    for (i=0; i<nelmts_send; i++) {
        int k = A->comm_handle->send_elmts_ids[i];
        PEVSL_CHKERR(k < 0 || k > A->ncol_local);
        A->comm_handle->send_buf[i] = x[k];
    }

    /* post non-blocking send */
    for (i=0; i<np_send; i++) {
        int p = A->comm_handle->proc_send_to[i];
        int j1 = A->comm_handle->send_elmts_ptr[i];
        int j2 = A->comm_handle->send_elmts_ptr[i+1];
        PEVSL_CHKERR(j2-j1 <= 0);
        MPI_Isend(A->comm_handle->send_buf+j1, j2-j1, MPI_DOUBLE, p, 0, \
                  A->comm, A->comm_handle->send_recv_requests+i);
    }
}

/**
 * @brief End function for matvecs
 * @param[in] A csr matrix
 */
void pEVSL_ParcsrMatvecCommEnd(pevsl_Parcsr *A) {
    int err;
    err = MPI_Waitall(A->comm_handle->num_proc_send_to + A->comm_handle->num_proc_recv_from, \
                      A->comm_handle->send_recv_requests, \
                      A->comm_handle->send_recv_status);
    PEVSL_CHKERR(err != MPI_SUCCESS);
}

/**
 * @brief (internal) Matrix vector product
 * @note Internal use only, use pEVSL_ParcsrMatvec
 * @param[in] x Input vector
 * @param[out] y output vector
 * @param[in] data csr matrix
 */
void pEVSL_ParcsrMatvec0(double *x, double *y, void *data) {
    pevsl_Parcsr *A = (pevsl_Parcsr *) data;

    pEVSL_ParcsrMatvecCommBegin(A, x);
    // overlapping computations with communications
    pEVSL_Matvec(A->diag, x, y);
    pEVSL_ParcsrMatvecCommEnd(A);
    if (A->offd->ncols > 0) {
        pEVSL_MatvecGen(1.0, A->offd, A->comm_handle->recv_buf, 1.0, y);
    }
}

/**
 * @brief Matrix vector product
 * @param[in] A csr matrix
 * @param[in] x Input vector
 * @param[out] y output vector
 */
void pEVSL_ParcsrMatvec(pevsl_Parcsr *A, pevsl_Parvec *x, pevsl_Parvec *y) {
  PEVSL_CHKERR(A->nrow_global != y->n_global);
  PEVSL_CHKERR(A->ncol_global != x->n_global);
  PEVSL_CHKERR(A->nrow_local  != y->n_local);
  PEVSL_CHKERR(A->ncol_local  != x->n_local);
  PEVSL_CHKERR(A->first_row   != y->n_first);
  PEVSL_CHKERR(A->first_col   != x->n_first);

  pEVSL_ParcsrMatvec0(x->data, y->data, (void *) A);
}

