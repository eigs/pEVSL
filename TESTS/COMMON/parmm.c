#include "pevsl.h"
#include "mmio.h"

/* @brief Read a matrix in Matrix-Market format in parallel
 * input fn   : file name
 *       idxin: 1: MM1 0: MM0
 *       row_col_starts:
 *       comm:
 * ouput A  : Parcsr matrix
 */ 
int ParcsrReadMM(pevsl_Parcsr *A, const char *fn, int idxin, int *row_col_starts, MPI_Comm comm) {
  int k, r1, r2, nrow, ncol, comm_size, comm_rank, nnz, sym;
  MM_typecode matcode;
  char line[MM_MAX_LINE_LENGTH];

  /* 0 or 1 */
  idxin = idxin > 0;

  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  FILE *p = fopen(fn, "r");
  if (p == NULL) {
    printf("Unable to open mat file %s\n", fn);
    exit(-1);
  }
  /*----------- READ MM banner */
  if (mm_read_banner(p, &matcode) != 0){
    printf("Could not process Matrix Market banner.\n");
    return 1;
  }
  if (!mm_is_valid(matcode)){
    printf("Invalid Matrix Market file.\n");
    return 1;
  }
  if ( !( (mm_is_real(matcode) || mm_is_integer(matcode)) && mm_is_coordinate(matcode) 
        && mm_is_sparse(matcode) ) ) {
    printf("Only sparse real-valued/integer coordinate matrices are supported\n");
    return 1;
  }
  /*------------- Read size */
  if (mm_read_mtx_crd_size(p, &nrow, &ncol, &nnz) !=0) {
    printf("MM read size error !\n");
    return 1;
  }
  //if (nrow != ncol) {
  //    fprintf(stdout,"This is not a square matrix!\n");
  //    return 1;
  //}

  /* parcsr matrix allocation */
  /* when row_starts == NULL, will use the default 1D partitioning */
  /* when col_starts == NULL, will use the default 1D partitioning */
  pEVSL_ParcsrCreate(nrow, ncol, row_col_starts, row_col_starts, A, comm);

  r1 = A->first_row;
  r2 = A->first_row + A->nrow_local;

  /*-------------------------------------------------
   * symmetric case : only L part stored in the file
   * need to recover the U part
   *------------------------------------------------*/
  if (mm_is_symmetric(matcode)){
    sym = 1;
  } else {
    sym = 0;
  }
  /*-------- Allocate mem for COO */
  int alloc_nnz = (nnz+comm_size-1) / comm_size * 2;
  int mynnz = 0, nnz_diag = 0;
  int *IR; 
  PEVSL_MALLOC(IR, alloc_nnz, int);
  int *JC; 
  PEVSL_MALLOC(JC, alloc_nnz, int);
  double *VV;
  PEVSL_MALLOC(VV, alloc_nnz, double);

  /*-------- read line by line */
  char *p1, *p2;
  for (k=0; k<nnz; k++) {
    if (fgets(line, MM_MAX_LINE_LENGTH, p) == NULL) {return -1;};
    for( p1 = line; ' ' == *p1; p1++ );
    /*----------------- 1st entry - row index */
    for( p2 = p1; ' ' != *p2; p2++ ); 
    *p2 = '\0';
    float tmp1 = atof(p1);
    int rid = (int) tmp1 - idxin;
    /*-------------- 2nd entry - column index */
    for( p1 = p2+1; ' ' == *p1; p1++ );
    for( p2 = p1; ' ' != *p2; p2++ );
    *p2 = '\0';
    float tmp2 = atof(p1);
    int cid = (int) tmp2 - idxin;
    /*------------- 3rd entry - nonzero entry */
    p1 = p2+1;
    double val = atof(p1);
    /*------------- */
    if (rid >= r1 && rid < r2) {
      if (mynnz == alloc_nnz) {
        alloc_nnz = alloc_nnz*2 + 1;
        PEVSL_REALLOC(IR, alloc_nnz, int);
        PEVSL_REALLOC(JC, alloc_nnz, int);
        PEVSL_REALLOC(VV, alloc_nnz, double);
      }
      PEVSL_CHKERR(mynnz >= alloc_nnz);
      IR[mynnz] = rid - r1;
      JC[mynnz] = cid;
      VV[mynnz] = val;
      mynnz++;
    }
    // symmetric matrix
    if (sym && rid != cid) {
      if (cid >= r1 && cid < r2) {
        if (mynnz == alloc_nnz) {
          alloc_nnz = alloc_nnz*2 + 1;
          PEVSL_REALLOC(IR, alloc_nnz, int);
          PEVSL_REALLOC(JC, alloc_nnz, int);
          PEVSL_REALLOC(VV, alloc_nnz, double);
        }
        PEVSL_CHKERR(mynnz >= alloc_nnz);
        IR[mynnz] = cid - r1;
        JC[mynnz] = rid;
        VV[mynnz] = val;
        mynnz++;
      }
    }
    if (rid == cid) {
      nnz_diag ++;
    }
  }
  fclose(p);

  int nnz_sum;
  MPI_Reduce(&mynnz, &nnz_sum, 1, MPI_INT, MPI_SUM, 0, comm);
  if (comm_rank == 0) {
    if (sym) {
      PEVSL_CHKERR(nnz_sum != 2*nnz-nnz_diag);
    } else {
      PEVSL_CHKERR(nnz_sum != nnz);
    }
  }

  pevsl_Coo coo_local;
  pevsl_Csr csr_local;

  coo_local.nrows = r2-r1;
  coo_local.ncols = ncol;
  coo_local.nnz = mynnz;
  coo_local.ir = IR;
  coo_local.jc = JC;
  coo_local.vv = VV;

  int ierr = pEVSL_CooToCsr(0, &coo_local, &csr_local); PEVSL_CHKERR(ierr);

  pEVSL_ParcsrSetup(&csr_local, A);

  pEVSL_FreeCoo(&coo_local);
  pEVSL_FreeCsr(&csr_local);

  return 0;
}

