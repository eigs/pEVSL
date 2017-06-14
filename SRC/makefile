.SUFFIXES : .f .c .o

include ../makefile.in

# Compilers
AR = ar rv
RANLIB = ranlib

INCLUDES = -I../INC 

# Object files
OBJS = cheblanNr.o chebpol.o comm.o lantrbnd.o pevsl.o miscla.o parcsr.o \
       parcsrmv.o parvec.o spmat.o stats.o utils.o vector.o \
       ../FORTRAN/pevsl_f90.o
       
LIB = ../libpevsl.a

# Rules
default: $(LIB)

%.f : %.o
	$(FORT) $(FLAGS) $(INCLUDES) -o $@ -c $<
%.o : %.c
	$(CC)   $(FLAGS) $(INCLUDES) -o $@ -c $<
$(LIB): $(OBJS)
	$(AR) $@ $(OBJS)
	$(RANLIB) $@

clean:
	rm -f $(OBJS) $(LIB)
