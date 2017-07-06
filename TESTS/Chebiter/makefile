.SUFFIXES : .c .o

include ../../makefile.in
include ../../EXTERNAL/makefile.in

#FLAGS ?= -DUNIX -O3 -g -Wall 

INCLUDES = -I../../INC -I../COMMON -I../../EXTERNAL $(MUMPS_INC)

# Object files
OBJS  = LapChebiter.o ../COMMON/comm.o ../COMMON/io.o ../COMMON/lapl.o ../../EXTERNAL/pevsl_mumps.o

LIB = -L../../ -lpevsl 

ALLEXE = 

LIB_EXT = $(LIBEXT) -fopenmp $(LIBLAPACK) $(LIB0) -lmpifort 

# intel Fortran lib
ifeq ($(LINK),mpiicc)
LIB_EXT += -lifcore
endif

# Rules
default: LapChebiter.ex

%.o : %.c
	$(CC) $(FLAGS) $(INCLUDES) -o $@ -c $<

# Exe
## polynomial
LapChebiter.ex: $(OBJS) ../../libpevsl.a
	$(LINK) -o LapChebiter.ex $(OBJS) $(LIB) $(LIB_EXT)

clean:
	rm -f *.o *.ex

cleanall:
	rm -f *.o *.ex OUT/*
