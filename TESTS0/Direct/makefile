.SUFFIXES : .c .o

include ../../makefile.in

INCLUDES = -I../../INC -I../../TESTS/COMMON -I../../EXTERNAL/DIRECTSOL

ifeq ($(INTEL), 1)
  OBJ_DIRSOL = ../../EXTERNAL/DIRECTSOL/pevsl_pardiso.o
else
  include ../../EXTERNAL/DIRECTSOL/makefile_mumps.in
  OBJ_DIRSOL = ../../EXTERNAL/DIRECTSOL/pevsl_mumps.o
  LIB_DIRSOL = $(LIB_MUMPS)
endif

# Object files
OBJS_CM = ../../TESTS/COMMON/comm.o ../../TESTS/COMMON/io.o ../../TESTS/COMMON/lapl.o
OBJS  = test.o $(OBJS_CM) $(OBJ_DIRSOL)

LIB = -L../../ -lpevsl 
LIB_EXT = $(LIB_DIRSOL) -fopenmp $(LIBLAPACK) $(LIB0)

# intel Fortran lib
ifeq ($(INTEL), 1)
LIB_EXT += -lifcore
else
LIB_EXT += -lmpifort
endif

# Rules
default: test.ex

%.o : %.c
	$(CC) $(FLAGS) $(INCLUDES) -o $@ -c $<

# Exe
test.ex: $(OBJS) ../../libpevsl.a
	$(LINK) -o test.ex $(OBJS) $(LIB) $(LIB_EXT)

clean:
	rm -f *.o *.ex

cleanall:
	rm -f *.o *.ex OUT/*
