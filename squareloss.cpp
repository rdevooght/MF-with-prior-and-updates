#include <vector>
#include <set>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include "squareloss.h"
#include "matrixfactorization.h"
#include "rowwisematrix.h"

SquareLoss::SquareLoss(SparseMatrix* _matrix, const int& InitComs, bool _Verbose) : MatrixFactorization(_matrix, InitComs, _Verbose)
{ 
	InitVSum();
	InitHSum();
}

void SquareLoss::InitVSum()
{
	// Initialization of SumV & matrixSumV to zero
	SumV = std::vector<double>(NumComs, 0);
	matrixSumV = std::vector<std::vector<double> >(NumComs, std::vector<double>(NumComs, 0));

	// Computation of SumV & matrixSumV
	for (int i = 0; i < V.size(); i++)
		AddToSum(matrixSumV, SumV, V[i]);

	/*
	for (int i = 0; i < NumComs; i++)
	{
		std::cerr << "SumV" << std::endl;
		for (int j = 0; j < NumComs; j++)
			std::cerr << matrixSumV[i][j] << " ";
		std::cerr << std::endl;
	}*/
}

void SquareLoss::InitHSum()
{
	// Initialization of SumH & matrixSumH to zero
	SumH = std::vector<double>(NumComs, 0);
	matrixSumH = std::vector<std::vector<double> >(NumComs, std::vector<double>(NumComs, 0));

	// Computation of SumH & matrixSumH
	for (int i = 0; i < H.size(); i++)
		AddToSum(matrixSumH, SumH, H[i]);

	/*
	for (int i = 0; i < NumComs; i++)
	{
		std::cerr << "SumH" << std::endl;
		for (int j = 0; j < NumComs; j++)
			std::cerr << matrixSumH[i][j] << " ";
		std::cerr << std::endl;
	}*/
}

void SquareLoss::ModifyNode(node nodeAddr, TIntFltH & rowFeatures)
{
	if (nodeAddr.orientation == ROW)
	{
		AddToSum(matrixSumV, SumV, V[nodeAddr.id], -1);
		V[nodeAddr.id] = rowFeatures;
		AddToSum(matrixSumV, SumV, V[nodeAddr.id]);
	}
	else
	{
		AddToSum(matrixSumH, SumH, H[nodeAddr.id], -1);
		H[nodeAddr.id] = rowFeatures;
		AddToSum(matrixSumH, SumH, H[nodeAddr.id]);
	}
}

double SquareLoss::ErrorForRow(const RowWiseMatrix::row* rowEntries, TIntFltH & rowFeatures, const std::vector<TIntFltH> & F, node currentNode)
{
	double L = 0.0;
	std::vector<double> productWithSumF;

	if (currentNode.orientation == ROW)
		MatrixVectorProduct(matrixSumH, rowFeatures, productWithSumF);
	else 
		MatrixVectorProduct(matrixSumV, rowFeatures, productWithSumF);

	L = NegWeight*DotProduct(rowFeatures, productWithSumF);

	for (int i = 0; i < rowEntries->size(); i++)
	{
		double dp = DotProduct(rowFeatures, F[(*rowEntries)[i].id]);
		L += (*rowEntries)[i].value * ((*rowEntries)[i].value - 2*dp) + (1 - NegWeight)*dp*dp;
	}

	//add regularization
	L += RegularizationPenaltyForRow(rowFeatures);

	if (L < 0)
	{
		std::cerr << currentNode.id << " " << currentNode.orientation << " loss: " << L << " (" << RegularizationPenaltyForRow(rowFeatures) << ") w=" << NegWeight << std::endl;
		std::cerr << "Matrix size " << matrix->n() << " " << matrix->m() << std::endl;
		throw std::runtime_error("Negative reconstruction error");
	}
	
	return L;
}

void SquareLoss::GradientForRow(const RowWiseMatrix::row* rowEntries, TIntFltH & rowFeatures, const std::vector<TIntFltH> & F, node currentNode, TIntFltH& gradient, const std::set<int>& CIDSet)
{
	std::vector<double> tempGradient(CIDSet.size(), 0.0);

	if (currentNode.orientation == ROW)
		PartialMatrixVectorProduct(matrixSumH, rowFeatures, tempGradient, CIDSet);
	else
		PartialMatrixVectorProduct(matrixSumV, rowFeatures, tempGradient, CIDSet);

	//account for factor 2 and weight of negative instances
	for (int i = 0; i < tempGradient.size(); ++i)
		tempGradient[i] *= 2*NegWeight;

	for (int i = 0; i < rowEntries->size(); i++)
	{
		int j = 0;
		double dp = DotProduct(rowFeatures, F[(*rowEntries)[i].id]);
		for (std::set<int>::iterator it=CIDSet.begin(); it!=CIDSet.end(); ++it, ++j)
			tempGradient[j] -= 2*( (*rowEntries)[i].value - (1 - NegWeight)*dp )*GetCom(F[(*rowEntries)[i].id], *it);
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
