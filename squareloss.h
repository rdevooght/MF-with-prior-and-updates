#ifndef square_loss
#define square_loss

#include <algorithm>
#include <set>
#include "matrixfactorization.h"
#include "rowwisematrix.h"


class SquareLoss : public MatrixFactorization
{	
	protected:
		std::vector<std::vector<double> > matrixSumV;
		std::vector<std::vector<double> > matrixSumH;

	public:
		SquareLoss(SparseMatrix* _matrix, const int& InitComs, bool _Verbose = false);
		virtual MatrixFactorization* Clone() { return new SquareLoss(*this); }
		virtual void Which() { std::cerr << "SquareLoss" << std::endl; }
		virtual double ErrorForRow(const RowWiseMatrix::row* rowEntries, TIntFltH & rowFeatures, const std::vector<TIntFltH> & F, node currentNode);
		virtual void GradientForRow(const RowWiseMatrix::row* rowEntries, TIntFltH & rowFeatures, const std::vector<TIntFltH> & F, node currentNode, TIntFltH& gradient, const std::set<int>& CIDSet);
		virtual void InitVSum();
		virtual void InitHSum();
		virtual void ModifyNode(node nodeAddr, TIntFltH & rowFeatures);

		inline void AddToSum(std::vector<std::vector<double> > & matrixSumF, std::vector<double> & SumF, const TIntFltH & rowFeatures, int sign = 1)
		{
			MatrixFactorization::AddToSum(SumF, rowFeatures, sign);
			for (TIntFltH::const_iterator it1 = rowFeatures.begin(); it1 != rowFeatures.end(); ++it1)
				for (TIntFltH::const_iterator it2 = rowFeatures.begin(); it2 != rowFeatures.end(); ++it2)
					matrixSumF[it1->first][it2->first] += sign * it1->second * it2->second;
		}

		inline void MatrixVectorProduct(const std::vector<std::vector<double> > & matrixSumF, const TIntFltH & rowFeatures, std::vector<double> & result)
		{
			int r = matrixSumF.size();
			result = std::vector<double>(r, 0);
			for (TIntFltH::const_iterator it = rowFeatures.begin(); it != rowFeatures.end(); it++)
				for (int i = 0; i < r; i++)
					result[i] += it->second * matrixSumF[it->first][i];
		}

		inline void PartialMatrixVectorProduct(const std::vector<std::vector<double> > & matrixSumF, const TIntFltH & rowFeatures, std::vector<double> & result, const std::set<int> & comSet)
		{
			int r = comSet.size();
			for (TIntFltH::const_iterator it = rowFeatures.begin(); it != rowFeatures.end(); it++)
			{
				int i = 0;
				for (std::set<int>::const_iterator setIt = comSet.begin(); setIt != comSet.end(); ++setIt, ++i)
					result[i] += it->second * matrixSumF[it->first][*setIt];
			}
		}

};

#endif
