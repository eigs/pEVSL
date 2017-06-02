.SUFFIXES : .c .o

include ../../makefile.in

FLAGS = -DUNIX -O3 -g -Wall 

MUMPSDIR = /home/ruipeng/workspace/MUMPS_5.1.1

INCLUDES = -I../../INC -I$(MUMPSDIR)/include

# Object files
OBJS  = test.o io.o lapl.o

LIB = -L../../ -lpevsl 

LIBMUMPS_COMMON = $(MUMPSDIR)/lib/libmumps_common.a
LIBDMUMPS = $(MUMPSDIR)/lib/libdmumps.a $(LIBMUMPS_COMMON)
LMETIS    = -L/home/ruipeng/workspace/parmetis-4.0.3/build/Linux-x86_64/libparmetis -lparmetis \
            -L/home/ruipeng/workspace/parmetis-4.0.3/build/Linux-x86_64/libmetis -lmetis
LPORD    = -L$(MUMPSDIR)/lib -lpord
LORDERINGS = $(LMETIS) $(LPORD)
LSCALAP  = /home/ruipeng/workspace/scalapack-2.0.2/libscalapack.a
#$(LAPACK)  -lmpi

LIB_EXT = -fopenmp $(LIBLAPACK) $(LIB0) -lmpifort

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
