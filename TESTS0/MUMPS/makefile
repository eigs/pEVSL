.SUFFIXES : .c .o

include ../../makefile.in

MUMPSDIR = /home/li50/workspace/MUMPS_5.1.1

INCLUDES = -I../../INC -I../../TESTS/COMMON -I$(MUMPSDIR)/include

# Object files
OBJS  = test.o ../../TESTS/COMMON/comm.o ../../TESTS/COMMON/io.o ../../TESTS/COMMON/lapl.o

LIB = -L../../ -lpevsl 

LIBMUMPS_COMMON = $(MUMPSDIR)/lib/libmumps_common.a
LIBDMUMPS = $(MUMPSDIR)/lib/libdmumps.a $(LIBMUMPS_COMMON)
LMETIS    = -L/home/li50/workspace/parmetis-4.0.3/build/Linux-x86_64/libparmetis -lparmetis \
            -L/home/li50/workspace/parmetis-4.0.3/build/Linux-x86_64/libmetis -lmetis
LPORD    = -L$(MUMPSDIR)/lib -lpord
LORDERINGS = $(LMETIS) $(LPORD)
LSCALAP  = /home/li50/workspace/scalapack-2.0.2/libscalapack.a
#$(LAPACK)  -lmpi

LIB_EXT = -fopenmp $(LIBLAPACK) $(LIB0) -lgfortran

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
	$(LINK) -o test.ex $(OBJS) $(LIB) $(LIBDMUMPS) $(LORDERINGS) $(LSCALAP) $(LIB_EXT)

clean:
	rm -f *.o *.ex

cleanall:
	rm -f *.o *.ex OUT/*
