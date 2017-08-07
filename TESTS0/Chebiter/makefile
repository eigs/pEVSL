.SUFFIXES : .c .o

include ../../makefile.in

INCLUDES = -I../../INC -I../../TESTS/COMMON -I../../EXTERNAL/ITERSOL

# Object files
OBJS = LapChebiter.o ../../TESTS/COMMON/comm.o ../../TESTS/COMMON/io.o \
       ../../TESTS/COMMON/lapl.o ../../EXTERNAL/ITERSOL/chebiter.o

LIB = -L../../ -lpevsl 

LIB_EXT = $(LIBLAPACK) $(LIB0)

# Rules
default: LapChebiter.ex

%.o : %.c
	$(CC) $(FLAGS) $(INCLUDES) -o $@ -c $<

# Exe
LapChebiter.ex: $(OBJS) ../../libpevsl.a
	$(LINK) -o LapChebiter.ex $(OBJS) $(LIB) $(LIB_EXT)

clean:
	rm -f *.o *.ex

cleanall:
	rm -f *.o *.ex OUT/*
