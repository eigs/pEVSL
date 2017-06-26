#include "pevsl_int.h"

void pEVSL_ParcsrMatvecCommBegin(pevsl_Parcsr *A, double *x) {
    int i;

    for (i=0; i<A->comm_handle->num_proc_recv_from; i++) {
        int p = A->comm_handle->proc_recv_from[i];
        int j1 = A->comm_handle->recv_elmts_ptr[i];
        int j2 = A->comm_handle->recv_elmts_ptr[i+1];
        PEVSL_CHKERR(j2-j1 <= 0);
        MPI_Irecv(A->comm_handle->recv_buf+j1, j2-j1, MPI_DOUBLE, p, 0, \
                  A->comm, A->comm_handle->recv_requests+i);
    }

    int np_send = A->comm_handle->num_proc_send_to;
    int nelmts_send = A->comm_handle->send_elmts_ptr[np_send];
    for (i=0; i<nelmts_send; i++) {
        int k = A->comm_handle->send_elmts_ids[i];
        PEVSL_CHKERR(k < 0 || k > A->ncol_local);
        A->comm_handle->send_buf[i] = x[k];
    }

    for (i=0; i<A->comm_handle->num_proc_send_to; i++) {
        int p = A->comm_handle->proc_send_to[i];
        int j1 = A->comm_handle->send_elmts_ptr[i];
        int j2 = A->comm_handle->send_elmts_ptr[i+1];
        PEVSL_CHKERR(j2-j1 <= 0);
        MPI_Isend(A->comm_handle->send_buf+j1, j2-j1, MPI_DOUBLE, p, 0, \
                  A->comm, A->comm_handle->send_requests+i);
    }
}

void pEVSL_ParcsrMatvecCommEnd(pevsl_Parcsr *A) {
    int err;
    err = MPI_Waitall(A->comm_handle->num_proc_send_to, A->comm_handle->send_requests, \
                A->comm_handle->send_status);
    PEVSL_CHKERR(err != MPI_SUCCESS);
    err = MPI_Waitall(A->comm_handle->num_proc_recv_from, A->comm_handle->recv_requests, \
                A->comm_handle->recv_status);
    PEVSL_CHKERR(err != MPI_SUCCESS);
}

/* @brief The most general form of ParcsrMatvec */
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

void pEVSL_ParcsrMatvec(pevsl_Parcsr *A, pevsl_Parvec *x, pevsl_Parvec *y) {
  pEVSL_ParcsrMatvec0(x->data, y->data, (void *) A);
}

