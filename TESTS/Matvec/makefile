.SUFFIXES : .c .o

include ../../makefile.in

#FLAGS = -DUNIX -O3 -g -Wall 

INCLUDES = -I../../INC -I../COMMON

# Object files
OBJS_1  = test1.o ../COMMON/comm.o ../COMMON/io.o ../COMMON/lapl.o
OBJS_2  = test2.o ../COMMON/comm.o ../COMMON/io.o ../COMMON/lapl.o
OBJS_3  = test3.o ../COMMON/comm.o ../COMMON/io.o ../COMMON/lapl.o
OBJS_4  = test4.o ../COMMON/comm.o ../COMMON/io.o ../COMMON/sprand.o

LIB = -L../../ -lpevsl 

ALLEXE = 

LIB_EXT = -fopenmp $(LIBLAPACK) $(LIB0)

# Rules
default: test1.ex test2.ex test3.ex test4.ex

%.o : %.c
	$(CC) $(FLAGS) $(INCLUDES) -o $@ -c $<

# Exe
test1.ex: $(OBJS_1) ../../libpevsl.a
	$(LINK) -o test1.ex $(OBJS_1) $(LIB) $(LIB_EXT)

test2.ex: $(OBJS_2) ../../libpevsl.a
	$(LINK) -o test2.ex $(OBJS_2) $(LIB) $(LIB_EXT)

test3.ex: $(OBJS_3) ../../libpevsl.a
	$(LINK) -o test3.ex $(OBJS_3) $(LIB) $(LIB_EXT)

test4.ex: $(OBJS_4) ../../libpevsl.a
	$(LINK) -o test4.ex $(OBJS_4) $(LIB) $(LIB_EXT)

clean:
	rm -f *.o *.ex

