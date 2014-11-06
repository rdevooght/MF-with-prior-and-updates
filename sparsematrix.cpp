#include "sparsematrix.h"
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>

SparseMatrix::SparseMatrix(unsigned long int numberOfRows, unsigned long int numberOfColumns) : rowWise(numberOfRows), colWise(numberOfColumns)
{}

SparseMatrix::SparseMatrix(const std::vector<int> & row_id, const std::vector<int> & col_id, const std::vector<double> & values) : rowWise(1), colWise(1)
{
	if (row_id.size() != col_id.size() || row_id.size() != values.size())
		throw std::runtime_error("SparseMatrix initialization: vectors' size does not match.");
	rowWise.Clear(*std::max_element(row_id.begin(), row_id.end()) + 1);
	colWise.Clear(*std::max_element(col_id.begin(), col_id.end()) + 1);
	
	for (int i = 0; i < row_id.size(); i++)
		SetEntry(row_id[i], col_id[i], values[i]);
}

// Initialize the matrix with nnz random non-zero elements
SparseMatrix::SparseMatrix(unsigned long int numberOfRows, unsigned long int numberOfColumns, unsigned long int nnz, double minValue, double maxValue) : rowWise(numberOfRows), colWise(numberOfColumns)
{
	unsigned long int initialized_nnz;

	boost::mt19937 Rnd(time(NULL));
	boost::mt19937 Rnd2(time(NULL)+1);
	boost::variate_generator<boost::mt19937, boost::uniform_int<> > rand_n(Rnd, boost::uniform_int<>(0, numberOfRows-1));
	boost::variate_generator<boost::mt19937, boost::uniform_int<> > rand_m(Rnd2, boost::uniform_int<>(0, numberOfColumns-1));
	srand(time(NULL));

	// first give a element to every rows and columns
	std::vector<unsigned long int> row_indices(numberOfRows);
	std::vector<unsigned long int> col_indices(numberOfColumns);

	for (unsigned long int i = 0; i < numberOfRows; ++i)
		row_indices[i] = i;
	for (unsigned long int i = 0; i < numberOfColumns; ++i)
		col_indices[i] = i;

	std::random_shuffle(row_indices.begin(), row_indices.end());
	std::random_shuffle(col_indices.begin(), col_indices.end());

	for (initialized_nnz = 0; initialized_nnz < std::min(nnz, std::max(numberOfColumns, numberOfRows)); ++initialized_nnz)
	{
		double value = double(rand()) / RAND_MAX * (maxValue - minValue) + minValue;
		SetEntry(row_indices[initialized_nnz % numberOfRows], col_indices[initialized_nnz % numberOfColumns], value); 
	}

	// Add random entries until nnz non-zero elements exist
	while (initialized_nnz < nnz)
	{
		unsigned long int i = rand_n();
		unsigned long int j = rand_m();
		double value = double(rand()) / RAND_MAX * (maxValue - minValue) + minValue;
		if (SetEntry(i, j, value))
			initialized_nnz++;
	}
}

void SparseMatrix::Clear(unsigned long int numberOfRows, unsigned long int numberOfColumns)
{
	rowWise.Clear(numberOfRows);
	colWise.Clear(numberOfColumns);
}

void SparseMatrix::LoadFromFile(const char* filename)
{
	//1st pass: see matrix size
	std::ifstream file(filename);
	int i, j, max_i = 0, max_j = 0;
	double value;
	while (file >> i >> j >> value)
	{
		if (i > max_i)
			max_i = i;
		if (j > max_j)
			max_j = j;
	}

	Clear(max_i+1, max_j+1);

	//2nd pass: add entries
	std::ifstream file2(filename);
	while (file2 >> i >> j >> value)
		SetEntry((unsigned long int) i, (unsigned long int) j, value);
}

RowWiseMatrix::row* SparseMatrix::GetRow(unsigned long int id)
{
	return rowWise.GetRow(id);
}

RowWiseMatrix::row* SparseMatrix::GetCol(unsigned long int id)
{
	return colWise.GetRow(id);
}

double SparseMatrix::Get(unsigned long int i, unsigned long int j)
{
	if (m() < n())
		return rowWise.Get(i,j);
	else return colWise.Get(j,i);
}

unsigned long int SparseMatrix::n()
{
	return rowWise.n();
}

unsigned long int SparseMatrix::m()
{
	return colWise.n();
}

bool SparseMatrix::SetEntry(unsigned long int i, unsigned long int j, double value)
{
	rowWise.SetEntry(i, j, value);
	return colWise.SetEntry(j, i, value);
}

void SparseMatrix::PrintAll()
{
	RowWiseMatrix::row* aRow;
	for (unsigned long int i = 0; i < n(); i++)
	{
		aRow = GetRow(i);
		for (unsigned long int j = 0; j < aRow->size(); j++)
			std::cout << i << " " << (*aRow)[j].id << " " << (*aRow)[j].value << std::endl;
	}
}