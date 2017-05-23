default:
	( cd SRC         ; make )
	( cd TESTS/Lap   ; mkdir -p OUT  ; make )

all:
	( cd SRC         ; make )
	( cd TESTS/Lap   ; mkdir -p OUT  ; make all)
	
clean:
	( cd SRC         ; make clean)
	( cd TESTS/Lap   ; make cleanall)


