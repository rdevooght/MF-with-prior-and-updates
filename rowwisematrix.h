#ifndef row_wise_matrix
#define row_wise_matrix

#include <vector>

class RowWiseMatrix {

	public:
		typedef struct {
			unsigned long int id;
			double value;
		} entry;

		typedef std::vector<entry> row;

		RowWiseMatrix(unsigned long int numberOfRows);
		~RowWiseMatrix();
		void Clear(unsigned long int numberOfRows = 1);
		row* GetRow(unsigned long int id);
		double Get(unsigned long int i, unsigned long int j);
		unsigned long int n(); // Get number of rows
		bool SetEntry(unsigned long int i, unsigned long int j, double value);

	protected:
		std::vector<row*> matrix;
};

#endif
