#include "pevsl_int.h"

/**
 * @file parcsr.c
 * @brief Functinos related to parallel csr structure
 * */
/*!
 * @brief create a parallel csr matrix struct
 * @param[in] nrow: global number of rows
 * @param[in] ncol: global number of columns
 * @param[in] row_starts: array of length NP + 1, which contains the first row index of each rank. NP = size of comm
 *                    row_starts[NP] should = nrow
 *                    if NULL, it means row_starts are from the ``default'' 1D partitioning,
 *                    see pEVSL_Part1d in utils.c for details
 * @param[in] col_starts: array of length NP + 1, which contains the first col index of each rank. NP = size of comm
 *                    col_starts[NP] should = ncol
 *                    if NULL, it means col_starts are from the ``default'' 1D partitioning,
 *                    see pEVSL_Part1d in utils.c for details
 * @param[out] A: parallel csr matrix
 * @param[in] comm: MPI communicator, the communicator that this matrix will reside on
 *
 * On return, A will not have any data but only information of sizes and row/col_starts
 * */
int pEVSL_ParcsrCreate(int nrow, int ncol, int *row_starts, int *col_starts, pevsl_Parcsr *A, MPI_Comm comm) {
  int r1, r2, c1, c2, comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  A->comm = comm;
  A->nrow_global = nrow;
  A->ncol_global = ncol;

  if (row_starts) {
    A->nrow_local = row_starts[comm_rank+1] - row_starts[comm_rank];
    PEVSL_MALLOC(A->row_starts, comm_size+1, int);
    memcpy(A->row_starts, row_starts, (comm_size+1)*sizeof(int));
    A->first_row = row_starts[comm_rank];
  } else {
    pEVSL_Part1d(nrow, comm_size, &comm_rank, &r1, &r2, 1);
    A->nrow_local = r2 - r1;
    A->row_starts = NULL;
    A->first_row = r1;
  }

  if (col_starts) {
    A->ncol_local = col_starts[comm_rank+1] - col_starts[comm_rank];
    PEVSL_MALLOC(A->col_starts, comm_size+1, int);
    memcpy(A->col_starts, col_starts, (comm_size+1)*sizeof(int));
    A->first_col = col_starts[comm_rank];
  } else {
    pEVSL_Part1d(ncol, comm_size, &comm_rank, &c1, &c2, 1);
    A->ncol_local = c2 - c1;
    A->col_starts = NULL;
    A->first_col = c1;
  }

  A->diag = NULL;
  A->offd = NULL;
  A->col_map_offd = NULL;
  A->comm_handle = NULL;

  return 0;
}

/*! @brief Construct a parcsr matrix from csr matrices.
 *  Each rank needs to provide its corresponding chunk of rows in a csr matrix
 *  @param[in] Ai: input the CSR matrix
 *  @param[out]  A: output the parallel CSR matrix
 */
int pEVSL_ParcsrSetup(pevsl_Csr *Ai, pevsl_Parcsr *A) {
  int i,j,k1,k2,c1,c2,nnz_Ai,nnz_diag,nnz_offd,ncol_offd,*work;
  int comm_size, comm_rank;
  MPI_Comm_size(A->comm, &comm_size);
  MPI_Comm_rank(A->comm, &comm_rank);

  nnz_Ai = Ai->ia[Ai->nrows];
  PEVSL_MALLOC(work, nnz_Ai, int);
  // nnz in diag and off-diag
  nnz_diag = nnz_offd = 0;
  // num of cols in off-diag [squeeze out zero columns]
  ncol_offd = 0;
  // [c1, c2) is the range of local columns
  c1 = A->first_col;
  c2 = c1 + A->ncol_local;
  // go through Ai for the first time, compute nnz in diag and off-diag,
  // and num of columns in off-diag, store all offd indices in work
  for (i=0; i<A->nrow_local; i++) {
    for (j=Ai->ia[i]; j<Ai->ia[i+1]; j++) {
      int col = Ai->ja[j];
      if (col >= c1 && col < c2) {
        nnz_diag++;
      } else {
        // store all offd col ind
        work[nnz_offd++] = col;
      }
    }
  }

  // sort these global col indices
  pEVSL_SortInt(work, nnz_offd);
  // after being sorted, this will find ncol_offd
  for (i=0; i<nnz_offd; i++) {
    if (i == 0 || work[i] != work[i-1]) {
      ncol_offd++;
    }
  }

  // allocate diag and offd csr's
  PEVSL_MALLOC(A->diag, 1, pevsl_Csr);
  pEVSL_CsrResize(A->nrow_local, A->ncol_local, nnz_diag, A->diag);
  PEVSL_MALLOC(A->offd, 1, pevsl_Csr);
  pEVSL_CsrResize(A->nrow_local, ncol_offd, nnz_offd, A->offd);
  PEVSL_MALLOC(A->col_map_offd, ncol_offd, int);

  /* build col_map_offd: this determines the order of columns of A->offd.
     Since work is sorted, col_map_offd is also sorted (in inreasing order).
     This is also important since we will use binary search later to determine their
     local indices
  */
  j = 0;
  for (i=0; i<nnz_offd; i++) {
    if (i == 0 || work[i] != work[i-1]) {
      A->col_map_offd[j++] = work[i];
    }
  }
  PEVSL_CHKERR(j != ncol_offd);

  // go through Ai for the 2nd time
  // split Ai into A->diag and A->offd
  // column indices in A->diag is changed to local ones
  // column indices in A->offd is changed to ``condensed'' indices
  A->diag->ia[0] = A->offd->ia[0] = 0;
  k1 = k2 = 0;
  for (i=0; i<A->nrow_local; i++) {
    for (j=Ai->ia[i]; j<Ai->ia[i+1]; j++) {
      int col = Ai->ja[j];
      double val = Ai->a[j];
      if (col >= c1 && col < c2) {
        /* adjust to local indices:
         * this is easily done by subtracting the starting col, c1 */
        A->diag->ja[k1] = col - c1;
        A->diag->a[k1] = val;
        k1++;
      } else {
        /* this is done by search where column index col is in col_map_offd */
        A->offd->ja[k2] = pEVSL_BinarySearch(A->col_map_offd, ncol_offd, col);
        /* this search must be successful */
        PEVSL_CHKERR(A->offd->ja[k2] < 0 || A->offd->ja[k2] >= ncol_offd);
        A->offd->a[k2] = val;
        k2++;
      }
    }
    A->diag->ia[i+1] = k1;
    A->offd->ia[i+1] = k2;
  }

  // build parCSR communication handle: for recv
  int np_recv = 0, *p_recv, *ptr_recv, *proc_offd;
  // temporarily store proc ids of each off-diag column
  PEVSL_MALLOC(proc_offd, ncol_offd, int);
  // go through col_map_offd for the first time to get np_recv
  for (i=0; i<ncol_offd; i++) {
    // global column index
    int col = A->col_map_offd[i];
    // will receive this data from proc p
    int p;
    if (!A->col_starts) {
      /* in the case of default partitioning */
      pEVSL_Part1d(A->ncol_global, comm_size, &p, &col, NULL, 2);
    } else {
      /* in the case of customized partitioning,
       * need to search for it */
      p = pEVSL_BinarySearchInterval(A->col_starts, comm_size+1, col);
    }
    /* p must be meaningful and not this rank */
    PEVSL_CHKERR(p < 0 || p >= comm_size || p == comm_rank);
    /* ``col'' needs to be received from rank p */
    proc_offd[i] = p;
    /* if this proc has not been seen before */
    if (i == 0 || p != proc_offd[i-1]) {
      np_recv ++;
    }
  }

  PEVSL_CHKERR(np_recv >= comm_size);

  PEVSL_MALLOC(A->comm_handle, 1, commHandle);
  PEVSL_MALLOC(p_recv, np_recv, int);
  PEVSL_MALLOC(ptr_recv, np_recv+1, int);

  // go through proc_offd to get p_recv and ptr_recv
  j = 0;
  for (i=0; i<ncol_offd; i++) {
    // will receive this data from proc p
    int p = proc_offd[i];
    if (i == 0 || p != proc_offd[i-1]) {
      p_recv[j] = p;
      ptr_recv[j++] = i;
    }
  }
  ptr_recv[np_recv] = ncol_offd;

  A->comm_handle->num_proc_recv_from = np_recv;
  A->comm_handle->proc_recv_from = p_recv;
  A->comm_handle->recv_elmts_ptr = ptr_recv;

  // exchange the recv data
  // number of elements to receive/send from each proc
  // dense vector [not scalable with nproc]
  int *num_elmts_recv, *num_elmts_send;
  PEVSL_CALLOC(num_elmts_recv, comm_size, int);
  PEVSL_MALLOC(num_elmts_send, comm_size, int);
  for (i=0; i<np_recv; i++) {
    num_elmts_recv[p_recv[i]] = ptr_recv[i+1]-ptr_recv[i];
    PEVSL_CHKERR(num_elmts_recv[p_recv[i]] <= 0);
  }
  //print_int_vec(comm->group_size, num_elmts_recv, comm);
  MPI_Alltoall(num_elmts_recv, 1, MPI_INT, num_elmts_send, 1, MPI_INT, A->comm);
  //print_int_vec(comm->group_size, num_elmts_send, comm);

  // build parCSR communication handle: send
  int np_send=0, *p_send, *elmts_send, nelmts_send=0, *ptr_send;
  for (i=0; i<comm_size; i++) {
    PEVSL_CHKERR(num_elmts_send[i] < 0);
    if (num_elmts_send[i]) {
      nelmts_send += num_elmts_send[i];
      np_send ++;
    }
  }
  PEVSL_MALLOC(p_send, np_send, int);
  PEVSL_MALLOC(ptr_send, np_send+1, int);
  j = 0;
  ptr_send[0] = 0;
  for (i=0; i<comm_size; i++) {
    if (num_elmts_send[i]) {
      p_send[j++] = i;
      ptr_send[j] = ptr_send[j-1] + num_elmts_send[i];
    }
  }
  PEVSL_CHKERR(j != np_send);
  PEVSL_CHKERR(ptr_send[j] != nelmts_send);

  int *elmts_recv = A->col_map_offd;
  int *elmts_recv_displs, *elmts_send_displs;
  PEVSL_MALLOC(elmts_send, nelmts_send, int);
  PEVSL_MALLOC(elmts_recv_displs, comm_size, int);
  PEVSL_MALLOC(elmts_send_displs, comm_size, int);
  elmts_recv_displs[0] = 0;
  elmts_send_displs[0] = 0;
  for (i=0; i<comm_size-1; i++) {
    elmts_recv_displs[i+1] = elmts_recv_displs[i] + num_elmts_recv[i];
    elmts_send_displs[i+1] = elmts_send_displs[i] + num_elmts_send[i];
  }
  MPI_Alltoallv(elmts_recv, num_elmts_recv, elmts_recv_displs, MPI_INT, \
                elmts_send, num_elmts_send, elmts_send_displs, MPI_INT, A->comm);

  //print_int_vec(ncol_offd, elmts_recv, comm);
  //print_int_vec(nelmts_send, elmts_send, comm);

  for (i=0; i<nelmts_send; i++) {
    //printf("%d: c1 %d c2 %d\n", elmts_send[i], c1, c2);
    PEVSL_CHKERR(!(elmts_send[i] >= c1 && elmts_send[i] < c2));
    elmts_send[i] -= c1;
  }

  A->comm_handle->num_proc_send_to = np_send;
  A->comm_handle->proc_send_to = p_send;
  A->comm_handle->send_elmts_ids = elmts_send;
  A->comm_handle->send_elmts_ptr = ptr_send;
  PEVSL_MALLOC(A->comm_handle->send_buf, nelmts_send, double);
  PEVSL_MALLOC(A->comm_handle->recv_buf, ncol_offd, double);
  PEVSL_MALLOC(A->comm_handle->send_requests, np_send, MPI_Request);
  PEVSL_MALLOC(A->comm_handle->recv_requests, np_recv, MPI_Request);
  PEVSL_MALLOC(A->comm_handle->send_status, np_send, MPI_Status);
  PEVSL_MALLOC(A->comm_handle->recv_status, np_recv, MPI_Status);

  PEVSL_FREE(proc_offd);
  PEVSL_FREE(num_elmts_recv);
  PEVSL_FREE(num_elmts_send);
  PEVSL_FREE(elmts_recv_displs);
  PEVSL_FREE(elmts_send_displs);
  PEVSL_FREE(work);

  return 0;
}


void pEVSL_ParcsrFree(pevsl_Parcsr *A) {
  PEVSL_FREE(A->comm_handle->proc_send_to);
  PEVSL_FREE(A->comm_handle->send_elmts_ids);
  PEVSL_FREE(A->comm_handle->send_elmts_ptr);
  PEVSL_FREE(A->comm_handle->proc_recv_from);
  PEVSL_FREE(A->comm_handle->recv_elmts_ptr);
  PEVSL_FREE(A->comm_handle->send_buf);
  PEVSL_FREE(A->comm_handle->recv_buf);
  PEVSL_FREE(A->comm_handle->send_requests);
  PEVSL_FREE(A->comm_handle->recv_requests);
  PEVSL_FREE(A->comm_handle->send_status);
  PEVSL_FREE(A->comm_handle->recv_status);
  PEVSL_FREE(A->comm_handle);
  PEVSL_FREE(A->row_starts);
  PEVSL_FREE(A->col_starts);
  pEVSL_FreeCsr(A->diag);
  PEVSL_FREE(A->diag);
  pEVSL_FreeCsr(A->offd);
  PEVSL_FREE(A->offd);
  PEVSL_FREE(A->col_map_offd);
}

/**
 * @brief extract the local matrix from Parcsr
 * local matrix of size m_local x N_global
 * Row indices: LOCAL, Col indices: GLOBAL
 *
 * @param[in] A Input matrix
 * @param[in] inx if matrix is 1 based
 * @param[out] coo local matrix in coo format
 * @param[out] csr local matrix in csr format
 * @param[in] stype 'L' or 'l', only return the "global" lower triangular portion
 *        'U' or 'u', only return the "global" upper triangular portion
 *        others    ,      return the "entire" local matrix
 * */
int pEVSL_ParcsrGetLocalMat(pevsl_Parcsr *A, int idx, pevsl_Coo *coo,
    pevsl_Csr *csr, char stype) {
  /* upper case letter */
  stype = toupper(stype);
  if (stype != 'L' && stype != 'U') {
    stype = 'A';
  }
  pevsl_Csr *Ad = A->diag;
  pevsl_Csr *Ao = A->offd;
  int local_nnz = PEVSL_CSRNNZ(Ad) + PEVSL_CSRNNZ(Ao);
  int i, j, k=0, nrow, ncol, fcol, frow;
  /* local csr matrix : slice of rows */
  pevsl_Csr csr_local;
  nrow = A->nrow_local;
  ncol = A->ncol_global;
  frow = A->first_row;
  fcol = A->first_col;
  pEVSL_CsrResize(nrow, ncol, local_nnz, &csr_local);
  csr_local.ia[0] = 0;
  /* for each row i */
  for (i=0; i<nrow; i++) {
    /* global row/col idx */
    int g_row, g_col;
    g_row = i + frow;
    /* diag block */
    for (j=Ad->ia[i]; j<Ad->ia[i+1]; j++) {
      /* global col idx */
      g_col = Ad->ja[j] + fcol;
      if ((stype == 'L' && g_row >= g_col) ||
          (stype == 'U' && g_row <= g_col) ||
          (stype == 'A'))
      {
        csr_local.ja[k] = g_col;
        csr_local.a[k] = Ad->a[j];
        k++;
      }
    }
    /* off-diag block */
    for (j=Ao->ia[i]; j<Ao->ia[i+1]; j++) {
      /* global col idx */
      g_col = A->col_map_offd[Ao->ja[j]];
      if ((stype == 'L' && g_row >= g_col) ||
          (stype == 'U' && g_row <= g_col) ||
          (stype == 'A'))
      {
        csr_local.ja[k] = g_col;
        csr_local.a[k] = Ao->a[j];
        k++;
      }
    }
    /* row ptr */
    csr_local.ia[i+1] = k;
  }

  PEVSL_CHKERR(k > local_nnz);
  PEVSL_CHKERR(stype == 'A' && k != local_nnz);

  /* sort row */
  pEVSL_SortRow(&csr_local);
  /* coo */
  if (coo) {
    pEVSL_CsrToCoo(&csr_local, idx, coo);
  }
  /* csr */
  if (csr) {
    /* adjust CSR index to 1-based */
    if (idx) {
      for (i=0; i<nrow; i++) {
        for (j=csr_local.ia[i]; j<csr_local.ia[i+1]; j++) {
          csr_local.ja[j]++;
        }
      }
      for (i=0; i<=nrow; i++) {
        csr_local.ia[i]++;
      }
    }
    *csr = csr_local;
  } else {
    pEVSL_FreeCsr(&csr_local);
  }

  return 0;
}

/**
 * @brief Returns number of GLOBAL non zeroes in A
 * @param[in] A Input matrix
 * @return number of GLOBAL non zeros
 * */
int pEVSL_ParcsrNnz(pevsl_Parcsr *A) {
  int nnz_global, nnz_local;
  pevsl_Csr *Ad = A->diag;
  pevsl_Csr *Ao = A->offd;
  nnz_local = PEVSL_CSRNNZ(Ad) + PEVSL_CSRNNZ(Ao);
  MPI_Allreduce(&nnz_local, &nnz_global, 1, MPI_INT, MPI_SUM, A->comm);

  return nnz_global;
}

/**
 * @brief Returns number of LOCAL non zeroes in A
 * @param[in] A Input matrix
 * @return number of LOCAL non zeroes
 * */
int pEVSL_ParcsrLocalNnz(pevsl_Parcsr *A) {
  pevsl_Csr *Ad = A->diag;
  pevsl_Csr *Ao = A->offd;
  int nnz_local = PEVSL_CSRNNZ(Ad) + PEVSL_CSRNNZ(Ao);

  return nnz_local;
}

