#include "rowwisematrix.h"
#include <cstddef>
#include <iostream>

RowWiseMatrix::RowWiseMatrix(unsigned long int numberOfRows) : matrix(numberOfRows, NULL)
{
	for (unsigned long int i = 0; i < numberOfRows; i++)
		matrix[i] = new row;
}

RowWiseMatrix::~RowWiseMatrix()
{
	for (unsigned long int i = 0; i < n(); i++)
		delete matrix[i];
}

void RowWiseMatrix::Clear(unsigned long int numberOfRows)
{
	for (unsigned long int i = 0; i < n(); i++)
		delete matrix[i];

	matrix = std::vector<row*>(numberOfRows, NULL);
	for (unsigned long int i = 0; i < numberOfRows; i++)
		matrix[i] = new row;
}

RowWiseMatrix::row* RowWiseMatrix::GetRow(unsigned long int id)
{
	if (id < n())
		return matrix[id];
	else return NULL;
}

double RowWiseMatrix::Get(unsigned long int i, unsigned long int j)
{
	// if i outside of range return 0
	if (i >= n())
		return 0;

	// look for j entry in ith row
	for (unsigned long int it = 0; it < matrix[i]->size(); it++)
		if ((*matrix[i])[it].id == j)
			return (*matrix[i])[it].value;
	
	// If entry wasn't found return 0
	return 0;
}

unsigned long int RowWiseMatrix::n()
{
	return matrix.size();
}

// Set entry (i,j) to value, return true if new entry was created, false if existing entry was modified
bool RowWiseMatrix::SetEntry(unsigned long int i, unsigned long int j, double value)
{
	// First step: expand matrix if needed
	if (i >= n())
	{
		unsigned long int missing_rows = i - n() + 1;
		for (unsigned long int it = 0; it < missing_rows; it++)
			matrix.push_back(new row);
	}

	// Second step: set entry
	for (unsigned long int it = 0; it < matrix[i]->size(); it++)
	{
		if ((*matrix[i])[it].id == j)
		{
			if (value == 0)
				matrix[i]->erase(matrix[i]->begin() + it);
			else (*matrix[i])[it].value = value;
			
			return false;
		}
	}
	if (value != 0)
	{
		entry newEntry = {j, value};
		matrix[i]->push_back(newEntry);
	}
	return true;
}