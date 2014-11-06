#include <algorithm>
#include <iostream>
#include <string>
#include <ctime>
#include <stdexcept>
#include "class/sparsematrix.h"
#include "class/absoluteloss.h"
#include "class/squareloss.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;


int main(int argc, char * argv[])
{
	try
	{
		unsigned long int n, m, nnz, r;
		double regularization_coef, alpha, beta, NegWeight;
		bool verbose, sl, al, abc;
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "produce help message")
			("file,f", po::value<std::string>(), "File with matrix description (format: 'row_id column_id value')")
			("rows,n", po::value<unsigned long int>(&n)->default_value(10), "number of rows")
			("columns,m", po::value<unsigned long int>(&m)->default_value(10), "number of columns")
			("non-zero-entries,e", po::value<unsigned long int>(&nnz)->default_value(20), "number of non-zero entries")
			("features,k", po::value<unsigned long int>(&r)->default_value(3), "number of features")
			("regularization-coef,r", po::value<double>(&regularization_coef)->default_value(0.0), "Regularization coefficient. Positive for L1, negative for L2.")
			("alpha,a", po::value<double>(&alpha)->default_value(0.3), "Alpha coeficient for line search.")
			("beta,b", po::value<double>(&beta)->default_value(0.3), "Beta coeficient for line search.")
			("negative-weight,w", po::value<double>(&NegWeight)->default_value(1), "Weight of negative (i.e. null) instances.")
			("verbose,v", po::value<bool>(&verbose)->implicit_value(true)->default_value(false), "enable verbosity")
			("sl", po::value<bool>(&sl)->implicit_value(true)->default_value(false), "Use square loss")
			("al", po::value<bool>(&al)->implicit_value(true)->default_value(false), "Use absolute loss")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		po::notify(vm);

		if (vm.count("help")) 
		{
			std::cout << desc;
			return 0;
		}
		
		// Generate random matrix
		SparseMatrix matrix(n, m, nnz);
		
		// Load matrix from file if a file is given
		if (vm.count("file"))
		{
			matrix.LoadFromFile(vm["file"].as<std::string>().c_str());
		}
		
		MatrixFactorization* optimizer;

		// Select loss function
		if (!al && !sl)
			throw std::runtime_error("Loss function unspecified");
		else if ((al && sl))
			throw std::runtime_error("Multiple loss functions specified");
		else
		{
			if (al)
				optimizer = new AbsoluteLoss(&matrix, r, verbose);
			else if (sl)
				optimizer = new SquareLoss(&matrix, r, verbose);
		}
		
		// Set options
		optimizer->SetRegCoef(regularization_coef);
		optimizer->SetAlpha(alpha);
		optimizer->SetBeta(beta);
		optimizer->SetNegWeight(NegWeight);
		
		// Optimize on all data
		optimizer->Optimize();
		
		// Print value of objective function
		std::cerr << optimizer->Error() << std::endl;
		
		// Print users' features (V) and items' features (H)
		optimizer->PrintV();
		optimizer->PrintH();
	}
	catch (const std::runtime_error& e)
	{
		std::cerr << e.what() << std::endl;
	}
	return 0;
}
