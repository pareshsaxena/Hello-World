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
#include <random>
#include <time.h>

#include "mmtsne.h"


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
	this->y_maps = y_maps;
	this->y_dims = y_dims;

	// Miscellaneous parameters
	double total_time = 0.0;
	clock_t start, end;

	switch (status) {
	case empty:
		std::cout << "No input vectors or probabilities loaded" << std::endl;
		return;	
	case input_vectors:
		std::cout << "Normalizing input vectors..." << std::endl;
		start = clock();
		// Normalize high-dimensional input vectors
		normalize(X);
		end = clock();
		std::cout << "\t Done. Time taken: " << std::setprecision(3) <<
			(double)(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
		std::cout << "Computing Stochastic Neighborhood Embedding..." << std::endl;
		start = clock();
		compute_SNE(P);
		end = clock();
		std::cout << "\t Done. Time taken: " << std::setprecision(3) <<
			(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;		
	case input_probability:		
		break;
	}
	
	// MM t-SNE data structures		
	std::vector<double> YD(x_rows * x_rows * y_maps, 0);			// Output pairwise distance matrix
	std::vector<double> Q(x_rows * x_rows, 0);						// Output pairwise similarity matrix			
	std::vector<double> Y_incs(x_rows * y_dims * y_maps, 0);		// Y increments matrix
	std::vector<double> W(x_rows * y_maps, 1.0 / y_maps);			// Weights matrix initialized with (1/ y_maps)
																	
	iW.resize(x_rows * y_maps);										// Resize importance weights matrix to size of W	
		
	std::vector<double> dCdY(x_rows * y_dims * y_maps, 0);			// Derivative of cost function w.r.t Y
	std::vector<double> dCdD(x_rows * x_rows * y_maps, 0);			// Derivative of cost function w.r.t Y distances
	std::vector<double> dCdW(x_rows * y_maps, 0);					// Derivative of cost function w.r.t W
	std::vector<double> dCdP(x_rows * y_maps, 0);					// Derivative of cost function w.r.t importance weights
	
	// Gradient descent parameters
	int max_iter = iterations;
	int stop_lying_iter = 25;
	int mom_switch_iter = 10;
	// Learning parameters
	double momentum = 0.5, final_momentum = 0.8;		
	
	// Lie about high-dimensional probabilities (P *= 4.0)	
	for (size_t i = 0; i < P.size(); ++i) {
		P[i] *= 4.0;
	}

	std::default_random_engine generator(time(NULL));
	std::normal_distribution<double> norm_dist(0.0, 1.0);			// Normal distribution with mean 0 & sigma 1.0

	// Initialize Y matrix with a random solution (values)
	Y.resize(x_rows * y_dims * y_maps);
	for (size_t i = 0; i < Y.size(); ++i) {
		Y[i] = norm_dist(generator) * 0.001;
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
		size_t x_rows2 = x_rows * x_rows;
		for (size_t m = 0; m < y_maps; ++m) {			
			for (size_t ri = 0; ri < x_rows; ++ri) {
				size_t base1_y = m * x_rows * y_dims + ri * y_dims;
				size_t base1_yd = m * x_rows2 + ri * x_rows;
				for (size_t rj = 0; rj < ri; ++rj) {
					size_t base2_y = m * x_rows * y_dims + rj * y_dims;
					size_t base2_yd = m * x_rows2 + rj * x_rows;
					double dist = 0;
					for (size_t c = 0; c < y_dims; ++c) {
						dist += pow((Y[base1_y + c] - Y[base2_y + c]), 2);						
					}
					YD[base1_yd + rj] = YD[base2_yd + ri] = (1.0 / (1.0 + dist));
				}
			}
		}			
						
		// Compute output pairwise similarity matrix Q
		compute_similarity_Q(Q, Z, YD);		

		// Compute error as Kullback-Liebler divergence between P & Q		
		if (verbose && iter % 1 == 0) {				
			/*long double sum_W = 0;
			for (size_t i = 0; i < W.size(); ++i) sum_W += W[i];
			std::cout << "\t Sum of Weights: " << sum_W << std::endl;*/
			double error = 0;
			error = compute_KL(P, Q);
			std::cout << "\t GD iteration: " << std::setw(4) << iter <<
				" | KL divergence (error): " << std::setprecision(15) << error << std::endl;
		}
		
		dCdD.assign(x_rows * x_rows * y_maps, 0);				// Reset dCdY[m] matrix elements to 0
		dCdY.assign(x_rows * y_dims * y_maps, 0);				// Reset dCdY matrix elements to 0
		// Compute Y gradients and update Y		
		std::thread Y_thread(&MMTSNE::Y_gradients, this, dCdY.data(), dCdD.data(),
			Y_incs.data(), P, Q, YD, Z, momentum);
		
		
		dCdP.assign(x_rows * y_maps, 0);						// Reset dCdP matrix elements to 0
		dCdW.assign(x_rows * y_maps, 0);						// Reset dCdW matrix elements to 0
		// Compute W gradients and update W
		std::thread W_thread(&MMTSNE::W_gradients, this, dCdW.data(), dCdP.data(), 
			W.data(), P, Q, YD, Z);		
		
		// Synchronize threads
		if (Y_thread.joinable()) Y_thread.join();
		if (W_thread.joinable()) W_thread.join();		

		// Update momentum 
		if (iter == mom_switch_iter) momentum = final_momentum;

		// It's about time we stopped lying about probabilites, dear... (P /= 4.0)		
		if (iter == stop_lying_iter) {
			for (size_t i = 0; i < P.size(); ++i) {
				P[i] /= 4.0;
			}			
		}
		
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
void MMTSNE::compute_similarity_Q(std::vector<double> &Q, double &Z, 
	const std::vector<double> &YD) {	
	// Initialize Q & Z
	Q.assign(x_rows * x_rows, DBL_MIN);
	Z = DBL_MIN;

	for (size_t ri = 0; ri < x_rows; ++ri) {
		size_t base1_q = ri * x_rows;
		size_t base1_iw = ri * y_maps;
		for (size_t rj = 0; rj < ri; ++rj) {
			size_t base2_q = rj * x_rows;
			size_t base2_iw = rj * y_maps;
			for (size_t m = 0; m < y_maps; ++m) {
				double qt = (iW[base1_iw + m] * iW[base2_iw + m] 
					* YD[m*x_rows*x_rows + base1_q + rj]);
				// DEBUG CODE
				if (isnan(qt)) {
					qt = DBL_MIN;
				}
				Q[base1_q + rj] += qt;
			}			
			Q[base2_q + ri] = Q[base1_q + rj];			
			Z += Q[base1_q + rj];
		}
	}
	Z *= 2.0;
	
	// Divide all elements in Q by Z		
	for (size_t ri = 0; ri < x_rows; ++ri) {
		for (size_t rj = 0; rj < ri; ++rj) {
			double qq = Q[ri*x_rows + rj] / Z;
			// DEBUG CODE
			if (isnan(qq) || qq < DBL_MIN) {
				Q[ri*x_rows + rj] = Q[rj*x_rows + ri] = DBL_MIN;
			}
			else {
				Q[ri*x_rows + rj] /= Z;
				Q[rj*x_rows + ri] = Q[ri*x_rows + rj];
			}
		}		
	}	
	
}


// Evaluate cost function as Kullback-Liebler divergence between P & Q
/* LaTex equation
C(Y) = KL(P||Q) = \sum_{r}\sum_{c\neq r} p_{rc} \: log \frac {p_{rc}}{q_{rc}}
*/
double MMTSNE::compute_KL(const std::vector<double> &P, const std::vector<double> &Q) {
	double kl_divergence = 0;

	for (size_t ri = 0; ri < x_rows; ++ri) {
		size_t base1 = ri * x_rows;
		for (size_t rj = 0; rj < ri; ++rj) {			
			kl_divergence += ((P[base1 + rj] < DBL_MIN ? DBL_MIN : P[base1 + rj]) *
				(log(P[base1 + rj] < DBL_MIN ? DBL_MIN : P[base1 + rj]) -
					log(Q[base1 + rj] < DBL_MIN ? DBL_MIN : Q[base1 + rj])));

			// DEBUG CODE
			if (isnan(kl_divergence) || isinf(kl_divergence)) {
				std::cout << "";
			}
		}
	}
	
	return kl_divergence;
}

// Compute Y gradients and update Y
void MMTSNE::Y_gradients(double *dCdY, double *dCdD, double *Y_incs, 
	const std::vector<double> &P, const std::vector<double> &Q, 
	const std::vector<double> &YD, const double &Z, const double& momentum) {
	double epsilonY = 5;							// Learning rate for Y (arbitrary)	
	
	// Compute dCdD
	/* LaTex equation
	\frac{\delta C(Y)}{\delta d_{rc}^{(m)}} = \frac{\pi_{r}^{(m)} \pi_{c}^{(m)} 
	(1+d_{rc}^{(m)})^{-1}} {q_{rc}Z}(p_{rc}-q_{rc})(1+d_{rc}^{(m)})^{-1}
	*/
	for (size_t m = 0; m < y_maps; ++m) {
		size_t base1 = m * x_rows * x_rows;
		for (size_t ri = 0; ri < x_rows; ++ri) {
			size_t base2 = ri * x_rows;
			size_t base2_iw = ri * y_maps;
			dCdD[base1 + base2 + ri] = iW[m*x_rows + ri] * iW[m*x_rows + ri];			
			for (size_t rj = 0; rj < ri; ++rj) {
				double qz = (Q[base2 + rj] * Z) < DBL_MIN ? DBL_MIN : (Q[base2 + rj] * Z);
				dCdD[base1 + rj*x_rows + ri] = dCdD[base1 + base2 + rj] = 
					 (iW[base2_iw + m] * iW[rj*y_maps + m] * (P[base2 + rj] - Q[base2 + rj]) *
					 pow(YD[base1 + base2 + rj], 2)) / (qz);
				
				// DEBUG CODE
				if (isnan(dCdD[base1 + rj*x_rows + ri]) || isinf(dCdD[base1 + rj*x_rows + ri])) {
					std::cout << "";
				}
			}
		}
	}

	// Compute dCdY
	/* LaTex equation
	\frac{\delta C(Y)}{\delta y_{r}^{(m)}} = 4\sum_{c} 
	\frac{\delta C(Y)}{\delta d_{rc}^{(m)}}(y_{r}^{(m)}-y_{c}^{(m)})
	*/	
	for (size_t m = 0; m < y_maps; ++m) {
		size_t base1_y = m * x_rows * y_dims;
		size_t base1_d = m * x_rows * x_rows;
		for (size_t ri = 0; ri < x_rows; ++ri) {
			size_t base2_y = ri * y_dims;
			size_t base2_d = ri * x_rows;
			for (size_t rj = 0; rj < x_rows; ++rj) {
				for (size_t d = 0; d < y_dims; ++d) {
					dCdY[base1_y + base2_y + d] += dCdD[base1_d + base2_d + rj] * 
						(Y[base1_y + base2_y + d] - Y[base1_y + rj*y_dims + d]);
				}				
			}			
		}
	}
	/*for (size_t i = 0; i < dCdY.size(); ++i) {
		dCdY[i] *= 4.0;
	}*/

	// Update Y
	for (size_t m = 0; m < y_maps; ++m) {
		size_t base1 = m * x_rows * y_dims;
		for (size_t ri = 0; ri < x_rows; ++ri) {
			size_t base2 = ri * y_dims;
			for (size_t d = 0; d < y_dims; ++d) {
				Y_incs[base1 + base2 + d] = momentum * Y_incs[base1 + base2 + d] -
					epsilonY * dCdY[base1 + base2 + d];
				Y[base1 + base2 + d] += Y_incs[base1 + base2 + d];
			}
		}
	}
	double mean = 0;
	for (size_t m = 0; m < y_maps; ++m) {
		size_t base1 = m * x_rows * y_dims;
		for (size_t d = 0; d < y_dims; ++d) {
			for (size_t r = 0; r < x_rows; ++r) {
				mean += Y[base1 + r*y_dims + d];
			}
			mean /= x_rows;
			for (size_t r = 0; r < x_rows; ++r) {
				Y[base1 + r*y_dims + d] -= mean;
			}
		}
	}
	
}

// Compute W gradients and update W
void MMTSNE::W_gradients(double *dCdW, double *dCdP, double *W,
	const std::vector<double> &P, const std::vector<double> &Q, 
	const std::vector<double> &YD, const double &Z) {
	double epsilonW = 2;							// Learning rate for W (arbitrary)
	// Compute dCdP
	/* LaTex equation
	\frac{\delta C(Y)}{\delta \pi_{i}^{(m)}} = \sum_{j}(\frac{2}{q_{ij}Z} 
	(p_{ij}-q_{ij})) \pi_{j}^{(m)} (1+\delta_{ij}^{m})^{-1} 
	*/
	
	for (size_t m = 0; m < y_maps; ++m) {		
		for (size_t ri = 0; ri < x_rows; ++ri) {
			size_t base1_p = ri * y_maps;
			size_t base1_s = ri * x_rows;
			for (size_t rj = 0; rj < x_rows; ++rj) {
				double qz = (Q[base1_s + rj] * Z) < DBL_MIN ? DBL_MIN : (Q[base1_s + rj] * Z);
				double dP = ((2 * (P[base1_s + rj] - Q[base1_s + rj]) *
					iW[base1_p + m] * YD[m*x_rows*x_rows + base1_s + rj]) / qz);
				
				// DEBUG CODE 
				if (isnan(dP) || isinf(dP)) {
					std::cout << "";
				}
				else dCdP[base1_p + m] += dP;
			}
		}
	}

	// Compute dCdW
	/* LaTex equation
	\frac{\delta C(Y)}{\delta w_{i}^{(m)}} = \pi_{i}^{(m)} ((\sum_{m'}\pi_{i}^{m'} 
	\frac{\delta C(Y)}{\delta \pi_{i}^{(m')}}) - \frac{\delta C(Y)}{\delta \pi_{i}^{(m)}}) 
	*/	
	for (size_t m = 0; m < y_maps; ++m) {
		for (size_t ri = 0; ri < x_rows; ++ri) {
			size_t base1 = ri * y_maps;
			for (size_t mk = 0; mk < y_maps; ++mk) {
				dCdW[base1 + mk] += (iW[base1 + mk] * dCdP[base1 + mk]);
			}
			dCdW[base1 + m] = iW[base1 + m] * (dCdW[base1 + m] - dCdP[base1 + m]);
			// DEBUG CODE
			if (isnan(dCdW[base1 + m])) {
				std::cout << "";
			}
		}
	}

	// Update W		
	for (size_t ri = 0; ri < x_rows; ++ri) {
		size_t base1 = ri * y_maps;
		for (size_t m = 0; m < y_maps; ++m) {
			if ((W[base1 + m] - (epsilonW * dCdW[base1 + m]))> 1) {
				W[base1 + m] = (W[base1 + m] - 1) * 0.5;		// Keep weights less than 1
			}
			else W[base1 + m] -= (epsilonW * dCdW[base1 + m]);

			// DEBUG CODE
			if (isnan(W[base1 + m])) {
				std::cout << "";
			}
		}
	}
}

// Symmetrize (square) matrix
void MMTSNE::symmetrize(std::vector<double> &M) {	
	double sum_M = 0;
	for (auto &mi : M) {
		sum_M += mi;
	}
	sum_M *= 2;

	for (size_t ri = 0; ri < x_rows; ++ri) {
		for (size_t rj = 0; rj < x_rows; ++rj) {
			M[ri*x_rows + rj] = M[rj*x_rows + ri] = (M[ri*x_rows + rj] + M[rj*x_rows + ri]) / sum_M;
		}
	}
}

// Update importance weights
/* LaTex equation
\pi _{i}^{(m)} = \frac {exp^{-w_{i}^{(m)}}} {\sum_{m'} exp^{-w_{i}^{(m')}}} 
*/
void MMTSNE::update_imp_W(const std::vector<double> &W) {	
	for (size_t ri = 0; ri < x_rows; ++ri) {
		double sum = 0;
		for (size_t m = 0; m < y_maps; ++m) {
			double xp = exp(W[ri*y_maps + m]);			
			if (isnan(xp) || isinf(xp)) {
				std::cout << "";
			}
			else {
				iW[ri*y_maps + m] = xp;
				sum += xp;
			}			
		}
		
		for (size_t m = 0; m < y_maps; ++m) {
			iW[ri*y_maps + m] /= sum;
			if (isnan(iW[ri*y_maps + m])) {
				std::cout << "";
			}
			/*else {
				iW[ri*y_maps + m] = DBL_MIN;					
			}*/
			
		}		
	}
}

// Compute the squared Euclidean distance matrix
void MMTSNE::compute_distance(const std::vector<double> &M, const size_t &dim, 
	std::vector<double> &DD) {
//	DD.assign(x_rows * x_rows, 0);
//	for (size_t ri = 0; ri < x_rows; ++ri) {
//		DD[ri*x_rows + ri] = 0;
//		for (size_t rj = 0; rj < ri; ++rj) {
//			for (size_t d = 0; d < dim; ++d) {
//				DD[ri*x_rows + rj] += pow(M[ri*x_rows + d] - M[rj*x_rows + d], 2);
//			}
//			DD[rj*x_rows + ri] = DD[ri*x_rows + rj];
//		}
//	}
}	

// Compute input similarities using a Gaussian kernel with a fixed perplexity
void MMTSNE::compute_Gaussian_kernel(const std::vector<double> &X_dist, std::vector<double> &P,
	size_t row_from, size_t row_to, size_t thread_id) {
	// Compute Gaussian kernel row by row
	clock_t start = clock();
	//for (size_t r = row_from; r < row_to; ++r) {
	//	// Initialize some variables		
	//	double beta = 1.0;
	//	double min_beta = -DBL_MAX, max_beta = DBL_MAX;
	//	double tol = 1e-5;
	//	double sum_P;

	//	// Iterate until a good perplexity is found using Binary Search
	//	for (size_t iter = 0; iter < 200; ++iter) {
	//		double H = 0;
	//		// Compute Gaussian kernel row
	//		for (size_t c = 0; c < P.cols(); ++c) {
	//			if (r == c) P(r, c) = DBL_MIN;
	//			else P(r, c) = exp(-beta * X_dist(r, c));
	//			H += beta * (X_dist(r, c) * P(r, c));
	//		}
	//		// Compute entropy of current row
	//		sum_P = P.row(r).sum() + DBL_MIN;
	//		H = (H / sum_P) + log(sum_P);

	//		// Evaluate whether the entropy is within the tolerance level
	//		double H_diff = abs(H - log(perplexity));
	//		if (H_diff < tol) {
	//			break;
	//		}
	//		else {
	//			if (H_diff > 0) {
	//				min_beta = beta;
	//				if (max_beta == DBL_MAX || max_beta == -DBL_MAX) beta *= 2.0;
	//				else beta = (beta + max_beta) / 2.0;
	//			}
	//			else {
	//				max_beta = beta;
	//				if (min_beta == -DBL_MAX || min_beta == DBL_MAX) beta /= 2.0;
	//				else beta = (beta + min_beta) / 2.0;
	//			}
	//		}
	//	}
	//	// Row normalize P
	//	for (size_t c = 0; c < P.cols(); ++c) P(r, c) /= sum_P;
	//}
	//clock_t end = clock();
	//if (verbose) {
	//	std::cout << "\t\tSNE | Thread #" << thread_id << " has ended | Time taken: " <<
	//		std::setprecision(3) << (end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
	//}
}

// Stochastic Neighborhood Embedding: convert high-dimensional Euclidean distance
// between data points into conditional probabilities that represent similarities
void MMTSNE::compute_SNE(std::vector<double> &P) {
	//std::vector<std::thread> thread_pool;
	//size_t max_threads = std::thread::hardware_concurrency() + 1;
	//size_t block = P.rows() / max_threads;

	//Eigen::MatrixXd X_dist(X.rows(), X.rows());
	//// Compute the squared Euclidean distance matrix
	//compute_distance(X, X_dist);

	//if (block == 0) {
	//	// Since (rows < max_threads) do not launch threads
	//	compute_Gaussian_kernel(X_dist, P, 0, P.rows(), 0);
	//}
	//else {
	//	for (size_t r = 0, thread_id = 1; r < P.rows(); r += block, ++thread_id) {			
	//		if (verbose) {
	//			std::cout << "\t\tSNE | Launching thread #" << thread_id << " | Rows [" << r <<
	//				", " << (((r + block) > P.rows())? P.rows() : (r + block)) << "]" << 
	//				std::endl;
	//		}
	//		thread_pool.push_back(std::thread(&MMTSNE::compute_Gaussian_kernel, this,
	//				X_dist, P, r, (((r + block) > P.rows()) ? P.rows() : (r + block)), thread_id));
	//	}
	//}
	//// Synchronize threads
	//for (auto &t : thread_pool) {
	//	if (t.joinable()) t.join();
	//}	
}

// Normalizes matrix (zero mean in the range [-1,1]
void MMTSNE::normalize(std::vector<double> &M) {
	// Matrix mean
	//double mean = M.mean();
	//
	//// Mean subtracted absolute max element in matrix
	//double abs_max = abs(M.maxCoeff() - mean);
	//double abs_min = abs(M.minCoeff() - mean);
	//double max = abs_max > abs_min ? abs_max : abs_min;
	//
	//// Normalize matrix
	//for (size_t r = 0; r < M.rows(); ++r) {
	//	for (size_t c = 0; c < M.cols(); ++c) {
	//		M[r, c] = (M[r,c] - mean) / max;
	//	}
	//}	
}



// Load high-dimensional input vector data from a CSV file
bool MMTSNE::load_input_vectors_csv(const std::string &fileName, const char &delimiter) {
	std::ifstream input_csv;
	input_csv.open(fileName);
	std::string line;	
	size_t rows = 0;
	size_t cols_prev = 0, cols = 0;
	clock_t start, end;

	std::cout << "Loading input vectors matrix..." << std::endl;

	start = clock();
	while (std::getline(input_csv, line)) {
		std::stringstream lineStream(line);
		std::string cell;
		cols = 0;
		while (std::getline(lineStream, cell, delimiter)) {
			try {
				X.push_back(std::stod(cell));
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
	end = clock();
	x_rows = rows;
	x_dims = cols;
		
	std::cout << "\t Done. Matrix of size: " << rows << " x " << cols
		<< " loaded." << "\n\t Time taken: " << std::setprecision(5) <<
		(double)(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;

	status = input_vectors;
}

// Load high-dimensional input probability data from a CSV file
bool MMTSNE::load_input_probability_csv(const std::string &fileName, 
	const char &delimiter) {
	std::ifstream input_csv;	
	std::string line;	
	size_t rows = 0;
	size_t cols_prev = 0, cols = 0;
	clock_t start, end;
	
	std::cout << "Loading input probability matrix..." << std::endl;

	start = clock();
	try {
		input_csv.open(fileName, std::ios::in);
		while (std::getline(input_csv, line)) {
			std::stringstream lineStream(line);
			std::string cell;
			cols = 0;
			while (std::getline(lineStream, cell, delimiter)) {
				try {
					P.push_back(std::stod(cell));
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
	end = clock();
	x_rows = rows;

	std::cout << "\t Done. Matrix of size: " << rows << " x " << cols
		<< " loaded." << "\n\t Time taken: " << std::setprecision(5) <<
		(double)(end - start) / CLOCKS_PER_SEC	<< " seconds" << std::endl;

	// TODO: Normalize P matrix
	double sum = 0;
	for (size_t ri = 0; ri < x_rows; ++ri) {
		double row_sum = 0;
		for (size_t rj = 0; rj < x_rows; ++rj) {
			row_sum += P[ri*x_rows + rj];
		}		
		for (size_t rj = 0; rj < x_rows; ++rj) {
			P[ri*x_rows + rj] /= row_sum;
			sum += P[ri*x_rows + rj];
		}
	}
	sum *= 2;
	for (size_t ri = 0; ri < x_rows; ++ri) {
		P[ri*x_rows + ri] = (P[ri*x_rows + ri] * 2) / sum;
		if (P[ri*x_rows + ri] < DBL_MIN) P[ri*x_rows + ri] = DBL_MIN;
		for (size_t rj = 0; rj < ri; ++rj) {
			P[ri*x_rows + rj] = (P[ri*x_rows + rj] + P[rj*x_rows + ri]) / sum;			
			if (P[ri*x_rows + rj] < DBL_MIN) {
				P[ri*x_rows + rj] = DBL_MIN;
				P[rj*x_rows + ri] = DBL_MIN;
			}
			else {
				P[rj*x_rows + ri] = P[ri*x_rows + rj];
			}

		}
	}

	status = input_probability;	
}

// Save low-dimensional output data to a CSV file
bool MMTSNE::save_output_vectors_csv(const std::string &fileName, const char &delimiter) {
	std::ofstream output_csv;
	try {
		output_csv.open(fileName, std::ios::out);
		for (size_t m = 0; m < y_maps; ++m) {
			size_t base1 = m * x_rows * y_dims;
			for (size_t r = 0; r < x_rows; ++r) {
				size_t base2 = r * y_dims;
				output_csv << (m + 1) << "," << iW[r*y_maps + m] << ",";
				for (size_t d = 0; d < y_dims; ++d) {
					output_csv << Y[base1 + base2 + d] << (d == (y_dims - 1) ? "\n" : ",");
				}
			}
		}
		output_csv.close();
		std::cout << "Output file saved" << std::endl;
		return true;
	}
	catch (std::exception &e) {
		std::cout << "Exception while saving CSV output file: " << e.what() << std::endl;
		if (output_csv.is_open()) output_csv.close();
		return false;
	}	
}
