CC=g++
OPT=-O2
CFLAGS=-c $(OPT)
LFLAGS=-lboost_program_options -L/usr/local/lib/ $(OPT)
EXECUTABLE=example
DIR=./
HEADER=$(DIR)absoluteloss.h $(DIR)squareloss.h $(DIR)matrixfactorization.h $(DIR)sparsematrix.h $(DIR)rowwisematrix.h
OBJECTS=absoluteloss.o squareloss.o matrixfactorization.o sparsematrix.o rowwisematrix.o

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) $(EXECUTABLE).o
	$(CC) $(LFLAGS) $(OBJECTS) $(EXECUTABLE).o -o $(EXECUTABLE)

$(EXECUTABLE).o: $(EXECUTABLE).cpp $(HEADER)
	$(CC) $(CFLAGS) $(EXECUTABLE).cpp

rowwisematrix.o: $(DIR)rowwisematrix.cpp $(DIR)rowwisematrix.h
	$(CC) $(CFLAGS) $(DIR)rowwisematrix.cpp

sparsematrix.o: $(DIR)sparsematrix.cpp $(DIR)sparsematrix.h $(DIR)rowwisematrix.h
	$(CC) $(CFLAGS) $(DIR)sparsematrix.cpp

matrixfactorization.o: $(DIR)matrixfactorization.cpp $(DIR)matrixfactorization.h $(DIR)sparsematrix.h $(DIR)rowwisematrix.h
	$(CC) $(CFLAGS) $(DIR)matrixfactorization.cpp

absoluteloss.o: $(DIR)absoluteloss.cpp $(DIR)absoluteloss.h $(DIR)matrixfactorization.h $(DIR)sparsematrix.h $(DIR)rowwisematrix.h
	$(CC) $(CFLAGS) $(DIR)absoluteloss.cpp

squareloss.o: $(DIR)squareloss.cpp $(DIR)squareloss.h $(DIR)matrixfactorization.h $(DIR)sparsematrix.h $(DIR)rowwisematrix.h
	$(CC) $(CFLAGS) $(DIR)squareloss.cpp

clean:
	rm -rf *o $(EXECUTABLE)
