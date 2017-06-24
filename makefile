all:
	( cd SRC           ; make )
	( cd EXTERNAL      ; make )
	( cd TESTS/Lap     ; mkdir -p OUT ; make)
	( cd TESTS/Gen_Lap ; mkdir -p OUT ; make)
	
clean:
	( cd SRC           ; make clean)
	( cd EXTERNAL      ; make clean)
	( cd TESTS/Lap     ; make clean)
	( cd TESTS/COMMON  ; make clean)
	( cd TESTS/Gen_Lap ; make clean)

