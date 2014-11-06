#ifndef sparse_matrix
#define sparse_matrix

#include "rowwisematrix.h"

class SparseMatrix { 
	protected:
		RowWiseMatrix rowWise;
		RowWiseMatrix colWise;

	public:
		SparseMatrix(unsigned long int numberOfRow = 1, unsigned long int numberOfColumns = 1);
		SparseMatrix(const std::vector<int> & row_id, const std::vector<int> & col_id, const std::vector<double> & values);
		SparseMatrix(unsigned long int numberOfRow, unsigned long int numberOfColumns, unsigned long int nnz, double minValue = 0, double maxValue = 10);
		void Clear(unsigned long int numberOfRows = 1, unsigned long int numberOfColumns = 1);
		void LoadFromFile(const char* filename);
		RowWiseMatrix::row* GetRow(unsigned long int id);
		RowWiseMatrix::row* GetCol(unsigned long int id);
		double Get(unsigned long int i, unsigned long int j);
		unsigned long int n(); // Get number of rows
		unsigned long int m(); // get number of columns
		bool SetEntry(unsigned long int i, unsigned long int j, double value);
		void PrintAll();
};

#endif
