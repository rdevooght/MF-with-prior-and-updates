
#include <iostream>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <set>
#include <cmath>
#include <stdexcept>
#include <boost/unordered_map.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include "rowwisematrix.h"
#include "matrixfactorization.h"

MatrixFactorization::MatrixFactorization(SparseMatrix* _matrix, const int& InitComs, bool _Verbose): 
matrix(_matrix), Rnd(time(NULL)), Verbose(_Verbose), RegCoef(0), MinVal(0.0), MaxVal(1000.0), MinGradNorm(1e-4), StoppingThreshold(1e-4), LSMaxIter(10), Alpha(0.3), Beta(0.3), NegWeight(1)
{
	srand(time(NULL));
	MaxIter = 10*(matrix->n()+matrix->m());
	RandomInit(InitComs);
}

void MatrixFactorization::SetMatrix(SparseMatrix* _matrix, const int& InitComs)
{
	matrix = _matrix;
	MaxIter = 10*(matrix->n()+matrix->m());
	RandomInit(InitComs);
}

void MatrixFactorization::SetMatrix(SparseMatrix* _matrix)
{
	SetMatrix(_matrix, NumComs);
}

void MatrixFactorization::SetNegWeight(const double _NegWeight) 
{
	if (_NegWeight < 0)
		throw std::runtime_error("NegWeight cannot be negative");
	NegWeight = _NegWeight;
}

void MatrixFactorization::RandomInit(const int InitComs)
{
	boost::mt19937 Rnd2(time(NULL)+1); // if n == m we need a second generator to avoid getting the same sequence of pseudo-random numbers
	boost::variate_generator<boost::mt19937, boost::uniform_int<> > rand_n(Rnd, boost::uniform_int<>(0, matrix->n() - 1));
	boost::variate_generator<boost::mt19937, boost::uniform_int<> > rand_m(Rnd2, boost::uniform_int<>(0, matrix->m() - 1));
	boost::variate_generator<boost::mt19937, boost::uniform_int<> > rand_k(Rnd, boost::uniform_int<>(0, InitComs - 1));

	V = std::vector<TIntFltH>(matrix->n());
	H = std::vector<TIntFltH>(matrix->m());
	std::vector<bool> used_feat(InitComs, false);
	NumComs = InitComs;

	for (int u = 0; u < V.size(); u++)
	{
		int Mem = matrix->GetRow(u)->size();
		if (Mem > 10) { Mem = 10; }
		for (int c = 0; c < Mem; c++)
		{
			int CID = rand_k();
			V[u][CID] = double(rand()) / RAND_MAX;
			used_feat[CID] = true;
		}

	}

	for (int u = 0; u < H.size(); u++)
	{
		int Mem = matrix->GetCol(u)->size();
		if (Mem > 10) { Mem = 10; }
		for (int c = 0; c < Mem; c++)
		{
			int CID = rand_k();
			H[u][CID] = double(rand()) / RAND_MAX;
		}
	}

	//assign a member to zero-member community (if any)
	for (int c = 0; c < NumComs; c++)
	{
		if (!used_feat[c])
		{
			unsigned long int i = rand_n();
			unsigned long int j = rand_m();
			V[i][c] = double(rand()) / RAND_MAX;
			H[j][c] = double(rand()) / RAND_MAX;
		}
	}

	InitVSum();
	InitHSum();
}

void MatrixFactorization::SetFactorization(const std::vector<TIntFltH> _V, const std::vector<TIntFltH> _H, const int NumberOfCommunities)
{
	V = _V;
	H = _H;
	NumComs = NumberOfCommunities;
	
	InitVSum();
	InitHSum();
}

void MatrixFactorization::ResizeV(unsigned int newMax)
{
	int rowsToAdd = newMax + 1 - V.size();
	V.resize(newMax + 1);
	for (;rowsToAdd > 0; rowsToAdd--)
	{
		//find least used feature
		int newFeature = std::distance(SumV.begin(), std::min_element(SumV.begin(), SumV.end()));

		// Add to current node
		TIntFltH newFeatures;
		newFeatures[newFeature] = 1;
		node newNode = {ROW, newMax + 1 - rowsToAdd};
		ModifyNode(newNode, newFeatures);
	}
}

void MatrixFactorization::ResizeH(unsigned int newMax)
{
	int rowsToAdd = newMax + 1 - H.size();
	H.resize(newMax + 1);
	for (;rowsToAdd > 0; rowsToAdd--)
	{
		//find least used feature
		int newFeature = std::distance(SumH.begin(), std::min_element(SumH.begin(), SumH.end()));

		// Add to current node
		TIntFltH newFeatures;
		newFeatures[newFeature] = 1;
		node newNode = {COLUMN, newMax + 1 - rowsToAdd};
		ModifyNode(newNode, newFeatures);
	}
}

void MatrixFactorization::PrintV()
{
	for (int i = 0; i < V.size(); i++)
		for (TIntFltH::iterator HI = V[i].begin(); HI != V[i].end(); ++HI)
			std::cout << i << " " << HI->first << " " << HI->second << std::endl;
}

void MatrixFactorization::PrintH()
{
	for (int i = 0; i < H.size(); i++)
		for (TIntFltH::iterator HI = H[i].begin(); HI != H[i].end(); ++HI)
			std::cout << i << " " << HI->first << " " << HI->second << std::endl;
}

void MatrixFactorization::PrintFeatures(const TIntFltH features)
{
	for (TIntFltH::const_iterator HI = features.begin(); HI != features.end(); ++HI)
		std::cout << "[" << HI->first << "," << HI->second << "] ";
	std::cout << std::endl;
}

// Initialize SumV, a vector that contains the sum of the rows of V
void MatrixFactorization::InitVSum()
{
	SumV = std::vector<double>(NumComs, 0);
	for (int i = 0; i < V.size(); i++)
		AddToSum(SumV, V[i]);
}

// Initialize SumV, a vector that contains the sum of the columns of H
void MatrixFactorization::InitHSum()
{
	SumH = std::vector<double>(NumComs, 0);
	for (int i = 0; i < H.size(); i++)
		AddToSum(SumH, H[i]);
}

// Change the features of the node nodeAddr, and update SumV or SumH accordingly
void MatrixFactorization::ModifyNode(node nodeAddr, TIntFltH & rowFeatures)
{
	if (nodeAddr.orientation == ROW)
	{
		AddToSum(SumV, V[nodeAddr.id], -1);
		V[nodeAddr.id] = rowFeatures;
		AddToSum(SumV, V[nodeAddr.id]);
	}
	else
	{
		AddToSum(SumH, H[nodeAddr.id], -1);
		H[nodeAddr.id] = rowFeatures;
		AddToSum(SumH, H[nodeAddr.id]);
	}
}

// Global reconstruction error
double MatrixFactorization::Error()
{ 
	double L = RegularizationPenalty(H);
	for (int i = 0; i < matrix->n(); i++)
		L += ErrorForRow(i);

	return L;
}

// The regularization penalty is specified by RegCoef. F is either V or H.
// If RegCoef == 0: no regularization
// If RegCoef > 0: penalty = RegCoef*L1_norm(F)
// If RegCoef < 0: penalty = |RegCoef|*L2_norm(F)
double MatrixFactorization::RegularizationPenalty(const std::vector<TIntFltH> & F)
{
	double result = 0;

	if (RegCoef > 0.0) //L1
		for (int i = 0; i < F.size(); i++)
			result += RegCoef * Sum(F[i]);
	
	if (RegCoef < 0.0) //L2
		for (int i = 0; i < F.size(); i++)
			result -= RegCoef * Norm2(F[i]);

	return result;
}

// Error associated with a given set of rows and columns. Used for the stopping criterion of the constrained optimization method.
double MatrixFactorization::LocalError(std::vector<int> & row_update_set, std::vector<int> & col_update_set)
{
	double local_error = 0;

	for (int i = 0; i < row_update_set.size(); ++i)
		local_error += ErrorForRow(row_update_set[i]);
	for (int i = 0; i < col_update_set.size(); ++i)
		local_error += ErrorForCol(col_update_set[i]);

	return local_error;
}


// Find the step size of the gradient step using the backtracking line search algorithm. (see Boyd, S. & Vandenberghe, L. (2009). Convex optimization)
double MatrixFactorization::LineSearch(const RowWiseMatrix::row* rowEntries, TIntFltH & rowFeatures, const std::vector<TIntFltH> & F, node currentNode, const TIntFltH& GradV)
{
	double StepSize = 1.0;
	double InitError = ErrorForRow(rowEntries, rowFeatures, F, currentNode);
	
	for(int iter = 0; iter < LSMaxIter; iter++)
	{
		// Build new features with current step size
		TIntFltH newFeatures;
		for (TIntFltH::const_iterator it = GradV.begin(); it != GradV.end(); ++it)
		{
			double NewVal = GetCom(rowFeatures, it->first) - StepSize * it->second;
			if (NewVal > MinVal) // limit features values between MinVal (0 for NMF) and MaxVal (by default 1000, to avoid absurd values)
			{
				if (NewVal > MaxVal) { NewVal = MaxVal; }
				newFeatures[it->first] = NewVal;
			}
		}

		// Check if new features are good enough. If not reduce step size
		if (ErrorForRow(rowEntries, newFeatures, F, currentNode) > InitError - Alpha * StepSize * Norm2(GradV))
			StepSize *= Beta;
		else break;
		
		// Too many iterations, return step size = 0
		if (iter == LSMaxIter - 1)
		{ 
			StepSize = 0.0;
			break;
		}
	}

	return StepSize;
}

int MatrixFactorization::ConstrainedOptimisation(int i, int j)
{
	std::vector<int> row_update_set, col_update_set;
	row_update_set.push_back(i);
	col_update_set.push_back(j);

	return ConstrainedOptimisation(row_update_set, col_update_set);
}

int MatrixFactorization::ConstrainedOptimisation(std::vector<int> & row_update_set, std::vector<int> & col_update_set)
{
	time_t InitTime = time(NULL);
	unsigned int iter = 0, PrevIter = 0;

	//Resize V and H if new rows or columns have to be optimized
	int max_row = *std::max_element(row_update_set.begin(), row_update_set.end());
	int max_col = *std::max_element(col_update_set.begin(), col_update_set.end());
	if (max_row > V.size() - 1)
		ResizeV(max_row);
	if (max_col > H.size() - 1)
		ResizeH(max_col);

	// Initilization of the shuffle vector.
	// The shuffle vector will contain all rows and columns indices and will determine in wich order the gradient steps are taken.
	// The vector is shuffled after each pass through the all set.
	std::vector<node> shuffle_vector;
	for (int i = 0; i < row_update_set.size(); i++) 
	{
		node nextNode = {ROW, row_update_set[i]};
		shuffle_vector.push_back(nextNode);
	}
	for (int i = 0; i < col_update_set.size(); i++)
	{
		node nextNode = {COLUMN, col_update_set[i]};
		shuffle_vector.push_back(nextNode);
	}

	double local_error = 0, prev_local_error = LocalError(row_update_set, col_update_set); // sum of the row Error of all the nodes within the update_set
		
	if (Verbose)
		std::cerr << "Starting optimization. Local error (E) = " << prev_local_error << std::endl;

	while(iter < MaxIter)
	{
		random_shuffle(shuffle_vector.begin(), shuffle_vector.end());
		
		// Iterate through all rows and columns
		for (int i = 0; i < shuffle_vector.size(); i++, iter++)
		{	
			if (shuffle_vector[i].orientation == ROW)
				stochasticGradientDescentStep(matrix->GetRow(shuffle_vector[i].id), V[shuffle_vector[i].id], H, shuffle_vector[i]);
			else
				stochasticGradientDescentStep(matrix->GetCol(shuffle_vector[i].id), H[shuffle_vector[i].id], V, shuffle_vector[i]);

			//AvoidTraps(shuffle_vector[i]);
		}
		// Evaluation of stopping criteria
		PrevIter = iter;
		local_error = LocalError(row_update_set, col_update_set);

		if (fabs(local_error - prev_local_error) <= StoppingThreshold*fabs(prev_local_error)) 
			break;
		else prev_local_error = local_error;

		if (Verbose)
			std::cerr << iter << " iterations [" << time(NULL) - InitTime << "sec] E = " << local_error << std::endl;
	}

	if (Verbose)
		std::cerr << "Optimization completed with " << iter << " iterations [" << time(NULL) - InitTime << "sec] E = " << local_error << std::endl;

	return iter;
}

void MatrixFactorization::stochasticGradientDescentStep(const RowWiseMatrix::row* rowEntries, TIntFltH newFeatures, const std::vector<TIntFltH> & F, node currentNode)
{
	TIntFltH GradV;
	std::set<int> comSet;

	double InitError = ErrorForRow(rowEntries, newFeatures, F, currentNode);
	int initFeat = newFeatures.size();

	// comSet will contain the indices of every communities/features that at least one neighbors of the current row has. Other communities/features can be set to zero 
	for (int i = 0; i < rowEntries->size(); i++)
	{
		for (TIntFltH::const_iterator CI = F[(*rowEntries)[i].id].begin(); CI != F[(*rowEntries)[i].id].end(); ++CI)
			comSet.insert(CI->first);
	}

	//remove the community membership which U does not share with its neighbors 
	for (TIntFltH::iterator CI = newFeatures.begin(); CI != newFeatures.end();)
	{
		if (comSet.count(CI->first) == 0)
			CI = newFeatures.erase(CI);
		else ++CI;
	}
	


	if (comSet.empty()) { return; }
	GradientForRow(rowEntries, newFeatures, F, currentNode, GradV, comSet);
	if (Norm2(GradV) < MinGradNorm) { return; }
	double LearnRate = LineSearch(rowEntries, newFeatures, F, currentNode, GradV);
	if (LearnRate == 0.0) { return; }

	// Update the features of the row
	for (TIntFltH::iterator it = GradV.begin(); it != GradV.end(); ++it)
	{
		double new_value = GetCom(newFeatures, it->first) - LearnRate * it->second;
		//std::cerr << it->first << " " << new_value << std::endl;
		if (new_value > MaxVal) { new_value = MaxVal; }
		if (new_value > MinVal)
			newFeatures[it->first] = new_value;
		else if (newFeatures.count(it->first) > 0)
			newFeatures.erase(it->first);
	}
	

	ModifyNode(currentNode, newFeatures);

	if (ErrorForRow(rowEntries, newFeatures, F, currentNode) > InitError)
		throw std::runtime_error("Error increased after");
}

// Optimize for every rows of V and every columns of H
int MatrixFactorization::Optimize()
{
	std::vector<int> row_update_set;
	for (int i = 0; i < matrix->n(); i++)
		row_update_set.push_back(i);
	std::vector<int> col_update_set;
	for (int i = 0; i < matrix->m(); i++)
		col_update_set.push_back(i);

	return ConstrainedOptimisation(row_update_set, col_update_set);
}

void MatrixFactorization::Epoch()
{
	// Initilization of the shuffle vector.
	// The shuffle vector will contain all rows and columns indices and will determine in wich order the gradient steps are taken.
	// The vector is shuffled after each pass through the all set.
	std::vector<node> shuffle_vector;
	for (int i = 0; i < matrix->n(); i++) 
	{
		node nextNode = {ROW, i};
		shuffle_vector.push_back(nextNode);
	}
	for (int i = 0; i < matrix->m(); i++)
	{
		node nextNode = {COLUMN, i};
		shuffle_vector.push_back(nextNode);
	}
	
	random_shuffle(shuffle_vector.begin(), shuffle_vector.end());
	// Iterate through all rows and columns
	for (int i = 0; i < shuffle_vector.size(); i++)
	{	
		if (shuffle_vector[i].orientation == ROW)
			stochasticGradientDescentStep(matrix->GetRow(shuffle_vector[i].id), V[shuffle_vector[i].id], H, shuffle_vector[i]);
		else
			stochasticGradientDescentStep(matrix->GetCol(shuffle_vector[i].id), H[shuffle_vector[i].id], V, shuffle_vector[i]);
	}
}
