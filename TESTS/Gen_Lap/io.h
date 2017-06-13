#ifndef IO_H
#define IO_H

/* types of user command-line input */
typedef enum {
  INT,
  DOUBLE,
  STR,
  NA
} ARG_TYPE;

int findarg(const char *argname, ARG_TYPE type, void *val, int argc, char **argv);

#endif
