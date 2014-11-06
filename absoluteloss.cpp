#include <set>
#include <vector>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include "absoluteloss.h"
#include "matrixfactorization.h"
#include "rowwisematrix.h"

AbsoluteLoss::AbsoluteLoss(SparseMatrix* _matrix, const int& InitComs, bool _Verbose) : MatrixFactorization(_matrix, InitComs, _Verbose)
{ }

double AbsoluteLoss::ErrorForRow(const RowWiseMatrix::row* rowEntries, TIntFltH & rowFeatures, const std::vector<TIntFltH> & F, node currentNode)
{
	double L = 0.0;

	if (currentNode.orientation == ROW)
		L = NegWeight*DotProduct(rowFeatures, SumH);
	else 
		L = NegWeight*DotProduct(rowFeatures, SumV);

	//std::cout << "dp with sum" << L << std::endl;

	for (int i = 0; i < rowEntries->size(); i++)
	{
		double prediction = DotProduct(rowFeatures, F[(*rowEntries)[i].id]);
		L += fabs((*rowEntries)[i].value - prediction) - NegWeight*prediction;
	}

	//add regularization
	L += RegularizationPenaltyForRow(rowFeatures);

	if (L < 0)
		throw std::runtime_error("Negative reconstruction error");
	
	return L;
}

void AbsoluteLoss::GradientForRow(const RowWiseMatrix::row* rowEntries, TIntFltH & rowFeatures, const std::vector<TIntFltH> & F, node currentNode, TIntFltH& gradient, const std::set<int>& CIDSet)
{
	std::vector<double> tempGradient;

	if (currentNode.orientation == ROW)
		for (std::set<int>::iterator it=CIDSet.begin(); it!=CIDSet.end(); ++it)
			tempGradient.push_back(NegWeight*SumH[*it]);
	else
		for (std::set<int>::iterator it=CIDSet.begin(); it!=CIDSet.end(); ++it)
			tempGradient.push_back(NegWeight*SumV[*it]);

	for (int i = 0; i < rowEntries->size(); i++)
	{
		double diff = (*rowEntries)[i].value - DotProduct(rowFeatures, F[(*rowEntries)[i].id]);
		int j = 0;
		if (diff > 0)
			for (std::set<int>::iterator it=CIDSet.begin(); it!=CIDSet.end(); ++it, ++j)
				tempGradient[j] -= (1+NegWeight)*GetCom(F[(*rowEntries)[i].id], *it);
		else if (diff == 0)
			for (std::set<int>::iterator it=CIDSet.begin(); it!=CIDSet.end(); ++it, ++j)
				tempGradient[j] -= NegWeight*GetCom(F[(*rowEntries)[i].id], *it);
		else
			for (std::set<int>::iterator it=CIDSet.begin(); it!=CIDSet.end(); ++it, ++j)
				tempGradient[j] += (1-NegWeight)*GetCom(F[(*rowEntries)[i].id], *it);
	}

	//add regularization
	RegularizationGradientForRow(tempGradient, CIDSet, rowFeatures);


	// Copy from vector storing to hashmap
	std::set<int>::iterator it = CIDSet.begin();
	for (int i = 0; i < tempGradient.size(); i++, ++it) 
	{
		//if (fabs(tempGradient[i]) < 0.0001) { continue; }
		gradient[*it] = tempGradient[i];
	}
}
