/* 
*
* MIT License
*
* Copyright (c) 2016, Paresh Saxena
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
*/


#include <cmath>
#include <exception>
#include <float.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <thread>
#include <vector>
#include <time.h>


#include "mmtsne.h"

#include <Eigen/Dense>


// Default constructor
MMTSNE::MMTSNE() : perplexity(30) {	
}


MMTSNE::MMTSNE(double new_perplexity) :
	perplexity(new_perplexity) {

}

// Destructor
MMTSNE::~MMTSNE() {}

// Perform Multiple Maps t-SNE (default settings)
void MMTSNE::construct_maps() {
	construct_maps(2, 30, 1000, true);
}

// Perform Multiple Maps t-SNE (verbose setting)
void MMTSNE::construct_maps(bool verbose) {
	construct_maps(2, 30, 1000, verbose);
}

// Perform Multiple Maps t-SNE (detailed settings)
void MMTSNE::construct_maps(int y_dims, int y_maps, int iterations, 
	bool verbose) {
	this->verbose = verbose;
	// Miscellaneous parameters
	float total_time = 0.0;
	clock_t start, end;

	switch (status) {
	case empty:
		std::cout << "No input vectors or probabilities loaded" << std::endl;
		break;	
	case input_vectors:
		std::cout << "Normalizing input vectors..." << std::endl;
		start = clock();
		// Normalize high-dimensional input vectors
		normalize(X);
		end = clock();
		std::cout << "\t Done. Time taken: " << std::setprecision(3) <<
			(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
		std::cout << "Computing Stochastic Neighborhood Embedding..." << std::endl;
		start = clock();
		compute_SNE(P);
		end = clock();
		std::cout << "\t Done. Time taken: " << std::setprecision(3) <<
			(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;		
	case input_probability:
		// Symmetrize input pairwise probability matrix
		symmetrize(P);
		break;
	}



	// MM t-SNE data structures		
	Eigen_Matrix3Xd YD = Matrix3Xd(X.rows(), X.rows(), y_maps, 0);			// Output pairwise distance matrix
	Eigen::MatrixXd Q  = Eigen::MatrixXd::Zero(X.rows(), X.rows());			// Output pairwise similarity matrix
	
	Eigen::MatrixXd W  = Eigen::MatrixXd::Constant(X.rows(), y_maps,
		1.0 / y_maps);														// Weights matrix initialized with (1/ y_maps)
	
	iW.resizeLike(W);														// Resize importance weights matrix to size of W

	
	Eigen_Matrix3Xd Y_incs = Matrix3Xd(X.rows(), y_dims, y_maps, 0);		// Y increments matrix
	
	Eigen_Matrix3Xd dCdY   = Matrix3Xd(X.rows(), y_dims, y_maps, 0);		// Derivative of cost function w.r.t Y
	Eigen::MatrixXd dCdW   = Eigen::MatrixXd::Zero(X.rows(), y_maps);		// Derivative of cost function w.r.t W
	
	// Gradient descent parameters
	int max_iter = 1000;
	int stop_lying_iter = max_iter / 4;
	int mom_switch_iter = max_iter / 4;
	// Learning parameters
	double momentum = 0.5, final_momentum = 0.8;
	double epsilonY = 250, epsilonW = 100;									// Learning rate for changes in Y & W

	// Lie about high-dimensional probabilities 
	P *= 4.0;

	// Initialize Y matrix with a random solution (values)
	for (size_t m = 0; m < y_maps; ++m) {
		(*Y[m]).setRandom();
		(*Y[m]) *= 0.001;
	}

	if (verbose) {
		std::cout << "Running gradient descent loop..." << std::endl;
		start = clock();
	}
	// Gradient descent loop
	for (size_t iter = 0; iter < max_iter; ++iter) {
		double Z = 0;
		// Update importance weights
		update_imp_W(W);

		// Compute output pairwise distance matrix YD
		/* LaTex equation
		YD_{rc} = YD_{cr} = (1 + ||y_{r} - y_{c}||)^{-1} 
		*/
		for (size_t m = 0; m < y_maps; ++m) {
			for (size_t r = 0; r < (*YD[m]).rows(); ++r) {
				(*YD[m])(r, r) = 1;
				for (size_t c = 0; c < r; ++c) {
					(*YD[m])(r, c) = 1 / (1 + ((*Y[m]).row(r) - (*Y[m]).row(c)).squaredNorm());
					(*YD[m])(c, r) = (*YD[m])(r, c);
				}
			}
		}
				
		// Compute output pairwise similarity matrix Q
		compute_similarity_Q(Q, Z, YD);

		// Compute error as Kullback-Liebler divergence between P & Q
		double error = 0;
		if (verbose && iter % 25 == 0) {
			error = compute_KL(P, Q);
			std::cout << "\t GD iteration: " << std::setw(4) << iter <<
				" | KL divergence (error): " << std::setprecision(6) << error << std::endl;
		}
				
		// Compute Y gradients
		std::thread Y_thread(&MMTSNE::Y_gradients, this, dCdY, P, Q, YD, Z);
		// Compute W gradients
		std::thread W_thread(&MMTSNE::W_gradients, this, dCdW, P, Q, YD, Z);

		// Synchronize threads
		if (Y_thread.joinable()) Y_thread.join();
		if (W_thread.joinable()) W_thread.join();

		// Update Y
		for (size_t m = 0; m < y_maps; ++m) {
			(*Y_incs[m]) = momentum * (*Y_incs[m]) - epsilonY * (*dCdY[m]);
			(*Y[m]) += (*Y_incs[m]);
		}

		// Update W
		W -= (epsilonW * dCdW);

		// Update momentum 
		if (iter == mom_switch_iter) momentum = final_momentum;

		// It's about time we stopped lying about probabilites, dear...
		if (iter == stop_lying_iter) P /= 4.0;
		
	}
	if (verbose) {
		end = clock();
		std::cout << "\t Done. Time taken: " << std::setprecision(3) <<
			(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
	}

}


// Compute output pairwise similarity matrix Q
/* LaTex equation
q_{rc} = \frac{\sum_{m}\pi_{r}^{(m)} \pi_{c}^{(m)} (1+d_{rc}^{(m)})^{-1}}
{\sum_{k}\sum_{l\neq k}\sum_{m'}\pi_{k}^{(m')} \pi_{l}^{(m')} (1+d_{kl}^{(m')})^{-1}}
*/
void MMTSNE::compute_similarity_Q(Eigen::MatrixXd &Q, double &Z, 
	const Eigen_Matrix3Xd &YD) {
	Q.setConstant(DBL_MIN);
	for (size_t r = 0; r < Q.rows(); ++r) {
		for (size_t c = 0; c < r; ++c) {
			for (size_t m = 0; m < y_maps; ++m) {
				Q(r, c) += iW(r, m) * iW(c, m) * (*YD[m])(r, c);
			}
			Q(c, r) = Q(r, c);
		}
	}
	Z = Q.sum();
	Q /= Z;
}

// Evaluate cost function as Kullback-Liebler divergence between P & Q
/* LaTex equation
C(Y) = KL(P||Q) = \sum_{r}\sum_{c\neq r} p_{rc} \: log \frac {p_{rc}}{q_{rc}}
*/
double MMTSNE::compute_KL(const Eigen::MatrixXd &P, const Eigen::MatrixXd &Q) {
	assert(P.rows() == Q.rows() && P.cols() == Q.cols());

	double kl_divergence = 0;
	for (size_t i = 0; i < P.rows(); ++i) {
		for (size_t j = 0; j < i; ++j) {
			kl_divergence += (P(i, j) * log(P(i, j) / Q(i, j)));
		}
	}
	
	return kl_divergence;
}

// Y gradients
void MMTSNE::Y_gradients(Eigen_Matrix3Xd &dCdY, const Eigen::MatrixXd &P,
	const Eigen::MatrixXd &Q, const Eigen_Matrix3Xd &YD, const double &Z) {																
	Eigen_Matrix3Xd dCdD = Matrix3Xd(X.rows(), X.rows(), y_maps, 0);			// Derivative of cost function w.r.t Y distances
	

	// Compute dCdD
	/* LaTex equation
	\frac{\delta C(Y)}{\delta d_{rc}^{(m)}} = \frac{\pi_{r}^{(m)} \pi_{c}^{(m)} 
	(1+d_{rc}^{(m)})^{-1}} {q_{rc}Z}(p_{rc}-q_{rc})(1+d_{rc}^{(m)})^{-1}
	*/
	for (size_t m = 0; m < y_maps; ++m) {
		for (size_t r = 0; r < (*dCdD[m]).rows(); ++r) {
			(*dCdD[m])(r, r) = iW(r, m) * iW(r, m);
			for (size_t c = 0; c < r; ++c) {
				(*dCdD[m])(r, c) = (*dCdD[m])(c, r) = (iW(r, m) * iW(c, m) * (P(r, c) - Q(r, c)) *
					(*YD[m])(r, c) * (*YD[m])(r, c)) / (Q(r, c) * Z);
			}
		}
	}

	// Compute dCdY
	/* LaTex equation
	\frac{\delta C(Y)}{\delta y_{r}^{(m)}} = 4\sum_{c} 
	\frac{\delta C(Y)}{\delta d_{rc}^{(m)}}(y_{r}^{(m)}-y_{c}^{(m)})
	*/
	for (size_t m = 0; m < y_maps; ++m) {
		(*dCdY[m]).setZero();													// Reset dCdY[m] matrix elements to 0
		for (size_t r = 0; r < (*dCdD[m]).rows(); ++r) {
			for (size_t c = 0; c < (*dCdD[m]).cols(); ++c) {
				(*dCdY[m]).row(r) += ((*dCdD[m])(r, c) * ((*Y[m]).row(r) - (*Y[m]).row(c)));
			}
			(*dCdY[m]).row(r) *= 4;
		}
	}

}

// W gradients
void MMTSNE::W_gradients(Eigen::MatrixXd &dCdW, const Eigen::MatrixXd &P,
	const Eigen::MatrixXd &Q, const Eigen_Matrix3Xd &YD, const double &Z) {
	Eigen::MatrixXd dCdP = Eigen::MatrixXd::Zero(X.rows(), y_maps);				// Derivative of cost function w.r.t importance weights
	
	// Compute dCdP
	/* LaTex equation
	\frac{\delta C(Y)}{\delta \pi_{r}^{(m)}} = \sum_{c}(\frac{2}{q_{rc}Z} 
	(p_{rc}-q_{rc})) \pi_{c}^{(m)} (1+\delta_{rc}^{m})^{-1} 
	*/
	for (size_t m = 0; m < y_maps; ++m) {
		for (size_t r = 0; r < dCdP.rows(); ++r) {
			for (size_t c = 0; c < dCdP.rows(); ++c) {
				dCdP(r, m) += ((2 * (P(r, c) - Q(r, c)) * iW(c, m) * (*YD[m])(r, c)) / (Q(r, c) * Z));
			}
		}
	}

	// Compute dCdW
	/* LaTex equation
	\frac{\delta C(Y)}{\delta w_{r}^{(m)}} = \pi_{r}^{(m)} ((\sum_{m'}\pi_{r}^{m'} 
	\frac{\delta C(Y)}{\delta \pi_{r}^{(m')}}) - \frac{\delta C(Y)}{\delta \pi_{r}^{(m)}}) 
	*/
	for (size_t m = 0; m < y_maps; ++m) {
		for (size_t r = 0; r < dCdW.rows(); ++r) {
			dCdW(r, m) = iW(r, m) * (((iW.row(r).array() * dCdP.row(r).array()).sum()) - dCdP(r, m));
		}
	}
}

// Symmetrize matrix
void MMTSNE::symmetrize(Eigen::MatrixXd &M) {	
	double sum_M = M.sum() * 2;
	for (size_t r = 0; r < M.rows(); ++r) {
		for (size_t c = 0; c < M.cols(); ++c) {
			M(r, c) = M(c, r) = (M(r, c) + M(c, r)) / sum_M;
		}
	}
}

// Update importance weights
/* LaTex equation
\pi _{i}^{(m)} = \frac {exp^{-w_{i}^{(m)}}} {\sum_{m'} exp^{-w_{i}^{(m')}}} 
*/
void MMTSNE::update_imp_W(const Eigen::MatrixXd &W) {	
	for (size_t r = 0; r < iW.rows(); ++r) {		
		for (size_t c = 0; c < iW.cols(); ++c) {
			iW(r,c) = exp(-W(r, c));
		}
		iW.row(r) /= iW.row(r).sum();
	}
}

// Compute the squared Euclidean distance matrix
void MMTSNE::compute_distance(const Eigen::MatrixXd &M, Eigen::MatrixXd &DD) {	
	for (size_t r = 0; r < M.rows(); ++r) {
		DD(r, r) = 0;
		for (size_t c = 0; c < r; ++c) {			
			DD(r, c) = DD(c,r) = (M.row(r) - M.row(c)).squaredNorm();
		}
	}
}	

// Compute input similarities using a Gaussian kernel with a fixed perplexity
void MMTSNE::compute_Gaussian_kernel(const Eigen::MatrixXd &X_dist, Eigen::MatrixXd &P,	
	size_t row_from, size_t row_to, size_t thread_id) {
	// Compute Gaussian kernel row by row
	clock_t start = clock();
	for (size_t r = row_from; r < row_to; ++r) {
		// Initialize some variables		
		double beta = 1.0;
		double min_beta = -DBL_MAX, max_beta = DBL_MAX;
		double tol = 1e-5;
		double sum_P;

		// Iterate until a good perplexity is found using Binary Search
		for (size_t iter = 0; iter < 200; ++iter) {
			double H = 0;
			// Compute Gaussian kernel row
			for (size_t c = 0; c < P.cols(); ++c) {
				if (r == c) P(r, c) = DBL_MIN;
				else P(r, c) = exp(-beta * X_dist(r, c));
				H += beta * (X_dist(r, c) * P(r, c));
			}
			// Compute entropy of current row
			sum_P = P.row(r).sum() + DBL_MIN;
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			double H_diff = abs(H - log(perplexity));
			if (H_diff < tol) {
				break;
			}
			else {
				if (H_diff > 0) {
					min_beta = beta;
					if (max_beta == DBL_MAX || max_beta == -DBL_MAX) beta *= 2.0;
					else beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if (min_beta == -DBL_MAX || min_beta == DBL_MAX) beta /= 2.0;
					else beta = (beta + min_beta) / 2.0;
				}
			}
		}
		// Row normalize P
		for (size_t c = 0; c < P.cols(); ++c) P(r, c) /= sum_P;
	}
	clock_t end = clock();
	if (verbose) {
		std::cout << "\t\tSNE | Thread #" << thread_id << " has ended | Time taken: " <<
			std::setprecision(3) << (end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
	}
}

// Stochastic Neighborhood Embedding: convert high-dimensional Euclidean distance
// between data points into conditional probabilities that represent similarities
void MMTSNE::compute_SNE(Eigen::MatrixXd &P) {
	std::vector<std::thread> thread_pool;
	size_t max_threads = std::thread::hardware_concurrency() + 1;
	size_t block = P.rows() / max_threads;

	Eigen::MatrixXd X_dist(X.rows(), X.rows());
	// Compute the squared Euclidean distance matrix
	compute_distance(X, X_dist);

	if (block == 0) {
		// Since (rows < max_threads) do not launch threads
		compute_Gaussian_kernel(X_dist, P, 0, P.rows(), 0);
	}
	else {
		for (size_t r = 0, thread_id = 1; r < P.rows(); r += block, ++thread_id) {			
			if (verbose) {
				std::cout << "\t\tSNE | Launching thread #" << thread_id << " | Rows [" << r <<
					", " << (((r + block) > P.rows())? P.rows() : (r + block)) << "]" << 
					std::endl;
			}
			thread_pool.push_back(std::thread(&MMTSNE::compute_Gaussian_kernel, this,
					X_dist, P, r, (((r + block) > P.rows()) ? P.rows() : (r + block)), thread_id));
		}
	}
	// Synchronize threads
	for (auto &t : thread_pool) {
		if (t.joinable()) t.join();
	}	
}

// Normalizes matrix (zero mean in the range [-1,1]
void MMTSNE::normalize(Eigen::MatrixXd &M) {
	// Matrix mean
	double mean = M.mean();
	
	// Mean subtracted absolute max element in matrix
	double abs_max = abs(M.maxCoeff() - mean);
	double abs_min = abs(M.minCoeff() - mean);
	double max = abs_max > abs_min ? abs_max : abs_min;
	
	// Normalize matrix
	for (size_t r = 0; r < M.rows(); ++r) {
		for (size_t c = 0; c < M.cols(); ++c) {
			M[r, c] = (M[r,c] - mean) / max;
		}
	}	
}


// Create 3D Eigen matrix initialized with 'value'
Eigen_Matrix3Xd MMTSNE::Matrix3Xd(const size_t& rows, const size_t& columns, 
	const size_t& depth, const double& value) {
	Eigen_Matrix3Xd M; 
	for (size_t d = 0; d < depth; ++d) {
		M.push_back(std::make_unique<Eigen::MatrixXd>(rows, columns));
		M[d]->setConstant(value);
	}

	return M;
}

// Load high-dimensional input vector data from a CSV file
bool MMTSNE::load_input_vectors_csv(const std::string &fileName, const char &delimiter) {
	std::ifstream input_csv;
	input_csv.open(fileName);
	std::string line;
	std::vector<double> values;
	size_t rows = 0;
	size_t cols_prev = 0, cols = 0;
	while (std::getline(input_csv, line)) {
		std::stringstream lineStream(line);
		std::string cell;
		cols = 0;
		while (std::getline(lineStream, cell, delimiter)) {
			try {
				values.push_back(std::stod(cell));
				++cols;
			}
			catch (std::exception &e) {
				std::cout << "Exception at row # " << rows << " | Value: " << 
					cell << std::endl;
				std::cout << " >> Message: " << e.what() << std::endl;
				return false;
			}			
		}
		++rows;
		if (cols_prev != cols && rows > 2) {			
			std::cout << "Exception at row # " << rows << ": Expecting " <<
				cols_prev << " columns but found " << cols << " columns" << std::endl;
			return false;
		}
		cols_prev = cols;
	}
	
	X = Eigen::Map<Eigen::Matrix<Eigen::MatrixXd::Scalar, Eigen::MatrixXd::RowsAtCompileTime,
		Eigen::MatrixXd::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, cols);
	
	std::cout << "Loaded input vector matrix of size: " << rows << " x " << cols
		<< std::endl;

	status = input_vectors;
	clock_t start, end;
	
}

// Load high-dimensional input probability data from a CSV file
bool MMTSNE::load_input_probability_csv(const std::string &fileName, const char &delimiter) {
	std::ifstream input_csv;
	
	std::string line;
	std::vector<double> values;
	size_t rows = 0;
	size_t cols_prev = 0, cols = 0;
	try {
		input_csv.open(fileName, std::ios::in);
		while (std::getline(input_csv, line)) {
			std::stringstream lineStream(line);
			std::string cell;
			cols = 0;
			while (std::getline(lineStream, cell, delimiter)) {
				try {
					values.push_back(std::stod(cell));
					++cols;
				}
				catch (std::exception &e) {
					std::cout << "Exception at row # " << rows << " | Value: " <<
						cell << std::endl;
					std::cout << " >> Message: " << e.what() << std::endl;
					return false;
				}
			}
			++rows;
			if (cols_prev != cols && rows > 2) {
				std::cout << "Exception at row # " << rows << ": Expecting " << 
					cols_prev << " columns but found " << cols << " columns" << std::endl;
				return false;
			}
			cols_prev = cols;
		}
		input_csv.close();
	}
	catch (std::exception &e) {
		std::cout << "Exception while reading input probability CSV file: " <<
			e.what() << std::endl;
		if (input_csv.is_open()) input_csv.close();
	}
	
	if (rows != cols) {
		std::cout << "Exception: Probability matrix is not square (rows != columns)" <<
			std::endl;
		return false;
	}

	P = Eigen::Map<Eigen::Matrix<Eigen::MatrixXd::Scalar, Eigen::MatrixXd::RowsAtCompileTime,
		Eigen::MatrixXd::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, cols);

	std::cout << "Loaded input vector matrix of size: " << rows << " x " << cols
		<< std::endl;

	status = input_probability;

	// TODO: Normalize P matrix
	
}

// Save low-dimensional output data to a CSV file
bool MMTSNE::save_output_vectors_csv(const std::string &fileName, const char &delimiter) {
	std::ofstream output_csv;
	try {
		output_csv.open(fileName, std::ios::out);
		for (size_t m = 0; m < y_maps; ++m) {
			for (size_t r = 0; r < (*Y[m]).rows(); ++r) {
				output_csv << (m + 1) << ",";
				for (size_t c = 0; c < (*Y[m]).cols(); ++c) {
					output_csv << (*Y[m])(r, c) << (c == ((*Y[m]).cols() - 1) ? "\n" : ",");
				}
			}
		}
		output_csv.close();
		std::cout << "Output file saved" << std::endl;
	}
	catch (std::exception &e) {
		std::cout << "Exception while saving CSV output file: " << e.what() << std::endl;
		if (output_csv.is_open()) output_csv.close();
	}	
}