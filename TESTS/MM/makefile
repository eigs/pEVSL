.SUFFIXES : .c .o

include ../../makefile.in

#FLAGS = -DUNIX -O3 -g -Wall 

INCLUDES = -I../../INC -I../COMMON

ifeq ($(INTEL), 1)
INCLUDES += -I${MKLROOT}/include
endif

# Object files
OBJS_PN  = MMPLanN.o ../COMMON/comm.o ../COMMON/io.o ../COMMON/parmm.o ../COMMON/mmio.o

LIB = -L../../ -lpevsl 

ALLEXE = 

LIB_EXT = -fopenmp $(LIBLAPACK) $(LIB0)

# Rules
default: MMPLanN.ex

%.o : %.c
	$(CC) $(FLAGS) $(INCLUDES) -o $@ -c $<

# Exe
## polynomial
MMPLanN.ex: $(OBJS_PN) ../../libpevsl.a
	$(LINK) -o MMPLanN.ex $(OBJS_PN) $(LIB) $(LIB_EXT)

clean:
	rm -f *.o *.ex

cleanall:
	rm -f *.o *.ex OUT/*
