all:
	( cd SRC         ; make )
	( cd EXTERNAL    ; make )
	( cd TESTS/Lap   ; mkdir -p OUT  ; make)
	
clean:
	( cd SRC         ; make clean)
	( cd EXTERNAL    ; make clean)
	( cd TESTS/Lap   ; make cleanall)


