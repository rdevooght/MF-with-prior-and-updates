
#ifndef matrix_factorization
#define matrix_factorization
#include <algorithm>
#include <vector>
#include <boost/unordered_map.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include "sparsematrix.h"
#include "rowwisematrix.h"

typedef enum {ROW, COLUMN} orientationType;

typedef struct {
	orientationType orientation;
	int id;
} node;

typedef boost::unordered_map<int, double> TIntFltH;

class MatrixFactorization { 
	protected:
		SparseMatrix* matrix;
		std::vector<TIntFltH> V; // Vertical part of factorization (Size: n * r)
		std::vector<TIntFltH> H; // Horizontal part of factorization (Size: r * m)
		std::vector<double> SumV; // Sum of the rows of V. Needed for efficient calculation
		std::vector<double> SumH; // Sum of the columns of H. Needed for efficient calculation
		boost::mt19937 Rnd; // random number generator
		double NegWeight; //Weight of negative instance (default: 1)
		double RegCoef; //Regularization coefficient when we fit for P_c +: L1, -: L2
		int NumComs; // number of communities
		double MinVal; // minimum value of F (0)
		double MaxVal; // maximum value of F (for numerical reason)
		double Alpha; // Parameter of line search
		double Beta; // Parameter of line search
		int LSMaxIter; // Parameter of line search
		int MaxIter; // Maximum number of iterations of the main optimization routine
		double StoppingThreshold;
		double MinGradNorm; // Doesn't perform a gradient step is gradient is inferior to MinGradNorm
		bool Verbose; // Print additional information if true

	public:
		MatrixFactorization(SparseMatrix* _matrix, const int& InitComs, bool _Verbose = false);
		virtual MatrixFactorization* Clone() = 0;
		virtual void Which() { std::cerr << "MatrixFactorization" << std::endl; }
		void SetMatrix(SparseMatrix* _matrix, const int& InitComs);
		void SetMatrix(SparseMatrix* _matrix);
		void SetRegCoef(const double _RegCoef) { RegCoef = _RegCoef; }
		double GetRegCoef() { return RegCoef; }
		void SetAlpha(const double _Alpha) { Alpha = _Alpha; }
		double GetAlpha() { return Alpha; }
		void SetBeta(const double _Beta) { Beta = _Beta; }
		double GetBeta() { return Beta; }
		void SetNegWeight(const double _NegWeight);
		double GetNegWeight() { return NegWeight; }
		void SetFactorization(const std::vector<TIntFltH> _V, const std::vector<TIntFltH> _H, const int NumberOfCommunities);
		void ResizeV(unsigned int newMax);
		void ResizeH(unsigned int newMax);
		std::vector<TIntFltH> GetV() { return V; };
		std::vector<TIntFltH> GetH() { return H; };
		std::vector<double> GetSumV() { return SumV; };
		std::vector<double> GetSumH() { return SumH; };
		void PrintV();
		void PrintH();
		void PrintFeatures(const TIntFltH features);
		void RandomInit(const int InitComs);
		double Error();
		double RegularizationPenalty(const std::vector<TIntFltH> & F);
		double LocalError(std::vector<int> & row_update_set, std::vector<int> & col_update_set);
		virtual double ErrorForRow(const RowWiseMatrix::row* rowEntries, TIntFltH & rowFeatures, const std::vector<TIntFltH> & F, node currentNode) = 0;
		virtual void GradientForRow(const RowWiseMatrix::row* rowEntries, TIntFltH & rowFeatures, const std::vector<TIntFltH> & F, node currentNode, TIntFltH& gradient, const std::set<int>& CIDSet) = 0;
		double LineSearch(const RowWiseMatrix::row* rowEntries, TIntFltH & rowFeatures, const std::vector<TIntFltH> & F, node currentNode, const TIntFltH& GradV);
		virtual int Optimize();
		void Epoch();
		void AvoidTraps(node currentNode);
		virtual int ConstrainedOptimisation(int i, int j);
		int ConstrainedOptimisation(std::vector<int> & row_update_set, std::vector<int> & col_update_set);
		void stochasticGradientDescentStep(const RowWiseMatrix::row* rowEntries, TIntFltH rowFeatures, const std::vector<TIntFltH> & F, node currentNode);
		virtual void InitVSum();
		virtual void InitHSum();
		virtual void ModifyNode(node nodeAddr, TIntFltH & rowFeatures);

		inline void AddToSum(std::vector<double> & SumF, const TIntFltH & rowFeatures, int sign = 1)
		{
			for (TIntFltH::const_iterator HI = rowFeatures.begin(); HI != rowFeatures.end(); ++HI)
				SumF[HI->first] += sign * HI->second;
		}

		double inline GetCom(const TIntFltH & rowFeatures, const int& CID)
		{
			if (rowFeatures.count(CID) > 0)
			{
				if (rowFeatures.at(CID) > 1e3)
					std::cerr << CID << " " << rowFeatures.at(CID) << std::endl;
				return rowFeatures.at(CID);
			}
			else
				return 0.0;
		}

		double inline Norm2(const TIntFltH& UV) 
		{
			double N = 0.0;
			for (TIntFltH::const_iterator HI = UV.begin(); HI != UV.end(); ++HI)
				N += HI->second * HI->second;
			return N;
		}

		double inline Sum(const TIntFltH& UV)
		{
			double N = 0.0;
			for (TIntFltH::const_iterator HI = UV.begin(); HI != UV.end(); ++HI)
				N += HI->second;
			return N;
		}

		double inline DotProduct(unsigned long int i, unsigned long int j)
		{
			if (i >= V.size() || j >= H.size())
				return 0;
			else return DotProduct(V[i], H[j]);
		}

		double inline DotProduct(const TIntFltH& UV, const TIntFltH& VV)
		{
			double DP = 0;
			if (UV.size() > VV.size()) {
				for (TIntFltH::const_iterator HI = UV.begin(); HI != UV.end(); ++HI) {
					if (VV.count(HI->first) > 0) { 
						DP += VV.at(HI->first) * HI->second; 
					}
				}
			} else {
				for (TIntFltH::const_iterator HI = VV.begin(); HI != VV.end(); ++HI) {
					if (UV.count(HI->first) > 0) { 
						DP += UV.at(HI->first) * HI->second; 
					}
				}
			}
			return DP;
		}

		double inline DotProduct(const TIntFltH& UV, const std::vector<double>& VV)
		{
			double DP = 0;
			
			for (TIntFltH::const_iterator HI = UV.begin(); HI != UV.end(); ++HI)
				if (HI->first < VV.size())
					DP += VV[HI->first] * HI->second;

			return DP;
		}

		inline double ErrorForRow(int rowId)
		{
			node currentNode = {ROW, rowId};
			return ErrorForRow(matrix->GetRow(rowId), V[rowId], H, currentNode);
		}

		inline double ErrorForCol(int colId)
		{
			node currentNode = {COLUMN, colId};
			return ErrorForRow(matrix->GetCol(colId), H[colId], V, currentNode);
		}

		inline double RegularizationPenaltyForRow(const TIntFltH & rowFeatures)
		{
			if (RegCoef > 0.0) //L1
				return RegCoef * Sum(rowFeatures);
			
			if (RegCoef < 0.0) //L2
				return - RegCoef * Norm2(rowFeatures);

			return 0.0;
		}

		inline void RegularizationGradientForRow(std::vector<double> & tempGradient, const std::set<int>& comSet, const TIntFltH & rowFeatures)
		{
			if (RegCoef > 0.0) //L1
				for (int i = 0; i < tempGradient.size(); i++)
					tempGradient[i] += RegCoef;
			if (RegCoef < 0.0) //L2
			{
				int i = 0;
				for (std::set<int>::const_iterator it=comSet.begin(); it!=comSet.end(); ++it, ++i)
					tempGradient[i] -= 2 * RegCoef * GetCom(rowFeatures, *it);
			}
		}
};

#endif
