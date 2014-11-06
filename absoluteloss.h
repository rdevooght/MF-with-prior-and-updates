#ifndef absolute_loss
#define absolute_loss
#include <algorithm>
#include <set>
#include "matrixfactorization.h"
#include "rowwisematrix.h"


class AbsoluteLoss : public MatrixFactorization
{	
	public:
		AbsoluteLoss(SparseMatrix* _matrix, const int& InitComs, bool _Verbose = false);
		virtual MatrixFactorization* Clone() { return new AbsoluteLoss(*this); }
		virtual void Which() { std::cerr << "AbsoluteLoss" << std::endl; }
		virtual double ErrorForRow(const RowWiseMatrix::row* rowEntries, TIntFltH & rowFeatures, const std::vector<TIntFltH> & F, node currentNode);
		virtual void GradientForRow(const RowWiseMatrix::row* rowEntries, TIntFltH & rowFeatures, const std::vector<TIntFltH> & F, node currentNode, TIntFltH& gradient, const std::set<int>& CIDSet);
};

#endif
