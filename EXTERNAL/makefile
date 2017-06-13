include ../makefile.in
include makefile.in

OBJ = pevsl_mumps.o 

INCLUDES = -I../INC -I. $(MUMPS_INC)

%.o : %.c
	$(CC) $(FLAGS) $(INCLUDES) -o $@ -c $<

all : $(OBJ)

clean:
	rm -f $(OBJ)

