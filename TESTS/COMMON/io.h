#ifndef IO_H
#define IO_H

#define MAX_LINE 1024
#define HB   1
#define MM0  2
#define MM1  3
#define UNKNOWN_FORMAT -1

/* types of user command-line input */
typedef enum {
  INT,
  DOUBLE,
  STR,
  NA
} ARG_TYPE;

int findarg(const char *argname, ARG_TYPE type, void *val, int argc, char **argv);

typedef struct _io_t {
  FILE  *fin;                  /* input FILE handle             */
  int    ntests;               /* number of test problems       */
  char   Fname1[MAX_LINE];     /* full matrix path name         */
  char   Fname2[MAX_LINE];     /* full matrix path name         */
  char   MatNam1[MAX_LINE];    /* short name                    */
  char   MatNam2[MAX_LINE];    /* short name                    */
  char   FmtStr[MAX_LINE];     /* matrix format string          */
  int    Fmt;                  /* matrix format type            */
  int    ndim;                 /* matrix size                   */
  int    nnz;                  /* number of nonzero             */
  int    nslices;              /* number of slices              */
  double a;                    /* [a, b] interval of interest  */
  double b;
} io_t;

int GetMatfileInfo(io_t *io, int job, const char *fname);

#endif
