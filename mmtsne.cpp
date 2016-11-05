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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <time.h>
#include <vector>

#include "mmtsne.h"


// Default constructor
MMTSNE::MMTSNE() : perplexity(30) {	
}

// Destructor
MMTSNE::~MMTSNE() {}

// Perform Multiple Maps t-SNE (default settings)
void MMTSNE::construct_maps() {
	construct_maps(2, 30, 1000, true);
}

// Perform Multiple Maps t-SNE (verbose setting)
void MMTSNE::construct_maps(bool verbose) {
	construct_maps(2, 5, 500, verbose);
}

// Perform Multiple Maps t-SNE (detailed settings)
void MMTSNE::construct_maps(size_t y_dims, size_t y_maps, size_t max_iter, bool verbose) {
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
	case input_probability:
		break;
	case input_vectors:
		std::cout << "Normalizing input vectors..." << std::endl;
		start = clock();
		// Normalize high-dimensional input vectors
		normalize(X, x_rows, x_dims);
		end = clock();
		std::cout << "\t Done. Time taken: " << std::setprecision(3) <<
			(double)(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
		std::cout << "Computing Stochastic Neighborhood Embedding..." << std::endl;
		start = clock();
		compute_SNE(P);
		end = clock();
		std::cout << "\t Done. Time taken: " << std::setprecision(3) <<
			(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
		break;	
	}
	
	// MM t-SNE data structures
	std::vector<double> YD(x_rows * x_rows * y_maps, 0);				// Output pairwise distance matrix
	std::vector<double> Q(x_rows * x_rows, 0);							// Output pairwise similarity matrix			
	
	std::vector<double> W(x_rows * y_maps, 1.0 / y_maps);				// Weights matrix initialized with (1/ y_maps)
																	
	iW.resize(x_rows * y_maps);											// Resize importance weights matrix to size of W	
	
	// Gradient descent data structures
	std::vector<double> dCdY(x_rows * y_dims * y_maps, 0);				// Derivative of cost function w.r.t Y
	std::vector<double> dCdD(x_rows * x_rows * y_maps, 0);				// Derivative of cost function w.r.t Y distances
	std::vector<double> dCdW(x_rows * y_maps, 0);						// Derivative of cost function w.r.t W
	std::vector<double> dCdP(x_rows * y_maps, 0);						// Derivative of cost function w.r.t importance weights

	std::vector<double> dCdY_exp(x_rows * y_dims * y_maps, 0);			// Exponential smoothing of the derivative of cost function w.r.t Y
	std::vector<double> dCdW_exp(x_rows * y_maps, 0);					// Exponential smoothing of the derivative of cost function w.r.t W

	std::vector<double> epsilon_Y(x_rows * y_dims * y_maps, 1);			// Learning rates for Y
	std::vector<double> epsilon_W(x_rows * y_maps, 1);					// Learning rates for W
		
	// Gradient descent parameters	
	int stop_lying_iter = (max_iter*0.05 > 30) ? 30 : (int)(max_iter*0.05);	
	
	double alpha = 0.65;												// Exponential smoothing parameter
	double epsilon_inc = 10; 											// Epsilon increment parameter (linear)
	double epsilon_dec = 0.55;											// Epsilon decrement parameter (exponential); should be in the range (0.1, 0.7]
		
	double alpha_update = (1 - alpha) / max_iter;						
	double epsilon_dec_update = (epsilon_dec - 0.1) / max_iter;

	double error = 0;
	double error_best = DBL_MAX;

	error_list.clear();
		
	// Lie about high-dimensional probabilities (P *= 4.0)	
	for (size_t i = 0; i < P.size(); ++i) {
		P[i] *= 4.0;
	}

	std::default_random_engine generator(time(NULL));
	std::normal_distribution<double> norm_dist(0.0, 1.0);				// Normal distribution with mean 0 & sigma 1.0

	// Initialize Y matrix with a random solution (values)
	Y.resize(x_rows * y_dims * y_maps);
	for (size_t i = 0; i < Y.size(); ++i) {
		Y[i] = norm_dist(generator);
	}
	
	if (verbose) {
		std::cout << "Running gradient descent loop..." << std::endl;
		start = clock();
	}

	// Gradient descent loop
	for (size_t iter = 0; iter < max_iter; ++iter) {
		// Z = \sum_{k} \sum_{l \neq k} \sum_{m'} \pi_{l}^{m'} \pi_{k}^{m'} (1 + d_{kl}^{m'})^{-1}
		double Z = 0;												
		
		// Update alpha
		alpha += alpha_update;

		// Update epsilon decrement parameter
		epsilon_dec -= epsilon_dec_update;

		// Update importance weights
		update_imp_W(W);

		// Compute output pairwise distance matrix YD
		/* LaTex equation
		YD_{ij} = YD_{ji} = (1 + ||y_{i} - y_{j}||)^{-1} 
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
		error = compute_KL(P, Q);
		error_list.push_back(error);
		if (error < error_best) {
			error_best = error;
			// Save the best solution
			Y_best = Y;
			iW_best = iW;
		}
		if (verbose) {			
			std::cout << "\t GD iteration: " << std::setw(4) << iter <<
				" | KL divergence (error): " << std::setprecision(15) << error << std::endl;			
		}
		
		dCdD.assign(x_rows * x_rows * y_maps, 0);						// Reset dCdY matrix elements to 0
		dCdY.assign(x_rows * y_dims * y_maps, 0);						// Reset dCdY matrix elements to 0
		// Compute Y gradients and update Y		
		std::thread Y_thread(&MMTSNE::Y_gradients, this, dCdY.data(), dCdY_exp.data(), 
			dCdD.data(), epsilon_Y.data(), P, Q, YD, Z, alpha, epsilon_inc, epsilon_dec);		
		
		dCdP.assign(x_rows * y_maps, 0);								// Reset dCdP matrix elements to 0
		dCdW.assign(x_rows * y_maps, 0);								// Reset dCdW matrix elements to 0
		// Compute W gradients and update W
		std::thread W_thread(&MMTSNE::W_gradients, this, dCdW.data(), dCdW_exp.data(),
			dCdP.data(), W.data(), epsilon_W.data(), P, Q, YD, Z, alpha, epsilon_inc, epsilon_dec);		
		
		// Synchronize threads
		if (Y_thread.joinable()) Y_thread.join();
		if (W_thread.joinable()) W_thread.join();				

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
			(end - start) / (CLOCKS_PER_SEC * 60) << " minutes\n" << std::endl;
		std::cout << "Best solution: KL divergence of " << std::setprecision(15) <<
			error_best << std::endl;		
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
				Q[base1_q + rj] += (iW[base1_iw + m] * iW[base2_iw + m]
					* YD[m*x_rows*x_rows + base1_q + rj]);				
				//if (!isnan(qt)) {
				//	Q[base1_q + rj] += qt;
				//}
				//else {
				//	std::cout << "";		// DEBUG CODE
				//}
			}			
			Q[base2_q + ri] = Q[base1_q + rj];			
			Z += Q[base1_q + rj];
		}
	}
	Z *= 2.0;
	
	// Divide all elements in Q by Z		
	for (size_t ri = 0; ri < x_rows; ++ri) {
		for (size_t rj = 0; rj < ri; ++rj) {
			Q[ri*x_rows + rj] /= Z;
			Q[rj*x_rows + ri] = Q[ri*x_rows + rj];
			//double qq = Q[ri*x_rows + rj] / Z;
			//// DEBUG CODE
			//if (isnan(qq)) {
			//	Q[ri*x_rows + rj] = Q[rj*x_rows + ri] = DBL_MIN;
			//}
			//else {
			//	Q[ri*x_rows + rj] /= Z;
			//	Q[rj*x_rows + ri] = Q[ri*x_rows + rj];
			//}
		}
	}

}


// Evaluate cost function as Kullback-Liebler divergence between P & Q
/* LaTex equation
C(Y) = KL(P||Q) = \sum_{i}\sum_{j\neq i} p_{ij} \: log \frac {p_{ij}}{q_{ij}}
*/
double MMTSNE::compute_KL(const std::vector<double> &P, const std::vector<double> &Q) {
	double kl_divergence = 0;

	for (size_t ri = 0; ri < x_rows; ++ri) {
		size_t base1 = ri * x_rows;
		for (size_t rj = 0; rj < ri; ++rj) {
			kl_divergence += (P[base1 + rj] *
				(log((P[base1 + rj] + DBL_MIN) / (Q[base1 + rj] + DBL_MIN))));
		}
	}

	return kl_divergence;
}

// Compute Y gradients and update Y
void MMTSNE::Y_gradients(double *dCdY, double *dCdY_exp, double *dCdD, double *epsilon_Y, 
	const std::vector<double> &P, const std::vector<double> &Q, const std::vector<double> &YD, 
	const double &Z, const double &alpha, const double &epsilon_inc, const double &epsilon_dec) {
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
				dCdD[base1 + rj*x_rows + ri] = dCdD[base1 + base2 + rj] =
					(iW[base2_iw + m] * iW[rj*y_maps + m] * (P[base2 + rj] - Q[base2 + rj]) *
						pow(YD[base1 + base2 + rj], 2)) / ((Q[base2 + rj] * Z) + DBL_MIN);

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

	// Update Y
	for (size_t m = 0; m < y_maps; ++m) {
		size_t base1 = m * x_rows * y_dims;
		for (size_t ri = 0; ri < x_rows; ++ri) {
			size_t base2 = ri * y_dims;
			for (size_t d = 0; d < y_dims; ++d) {
				double delta = dCdY[base1 + base2 + d] * dCdY_exp[base1 + base2 + d];
				if (delta > 0) {
					// Increment epsilon linearly
					epsilon_Y[base1 + base2 + d] += epsilon_inc;
				}
				else {
					// Decrement epsilon exponentially
					epsilon_Y[base1 + base2 + d] *= epsilon_dec;
				}

				// Exponential smoothing of dCdY using alpha (multiply dCdY by 4 as per gradient equation)
				dCdY_exp[base1 + base2 + d] = (alpha * 4 * dCdY[base1 + base2 + d]) +		
					((1 - alpha) * dCdY_exp[base1 + base2 + d]); 

				// Update Y
				Y[base1 + base2 + d] -= (epsilon_Y[base1 + base2 + d] *
					dCdY_exp[base1 + base2 + d]);
			}
		}
	}

	// Make Y zero mean along all dimensions
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
void MMTSNE::W_gradients(double *dCdW, double *dCdW_exp, double *dCdP, double *W,
	double *epsilon_W, const std::vector<double> &P, const std::vector<double> &Q,
	const std::vector<double> &YD, const double &Z, const double &alpha,
	const double &epsilon_inc, const double &epsilon_dec) {
	
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
				dCdP[base1_p + m] += ((2 * (P[base1_s + rj] - Q[base1_s + rj]) *
					iW[base1_p + m] * YD[m*x_rows*x_rows + base1_s + rj]) / 
					((Q[base1_s + rj] * Z) + DBL_MIN));				
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
		}
	}
	bool reset = false;
	// Update W		
	for (size_t ri = 0; ri < x_rows; ++ri) {
		size_t base1 = ri * y_maps;
		double max_w = DBL_MIN;
		for (size_t m = 0; m < y_maps; ++m) {
			double delta = dCdW[base1 + m] * dCdW_exp[base1 + m];
			if (delta > 0) {
				epsilon_W[base1 + m] += epsilon_inc;
			}
			else {
				epsilon_W[base1 + m] *= epsilon_dec;
			}

			// Exponential smoothing of dCdW
			dCdW_exp[base1 + m] = (alpha * dCdW[base1 + m]) +
				((1 - alpha) * dCdW_exp[base1 + m]);

			W[base1 + m] -= (epsilon_W[base1 + m] * dCdW[base1 + m]);
			if (W[base1 + m] > max_w) max_w = W[base1 + m];

		}

		// Keep weights negative
		for (size_t m = 0; m < y_maps; ++m) {
			W[base1 + m] -= max_w;
			if (W[base1 + m] < -1e20) {
				reset = true;
				break;
			}
		}		
	}
	if (reset) {
		for (size_t r = 0; r < x_rows; ++r) {
			double max_w = DBL_MIN;
			for (size_t c = 0; c < y_maps; ++c)
				if (dCdW_exp[r*y_maps + c] > max_w) max_w = dCdW_exp[r*y_maps + c];
			for (size_t c = 0; c < y_maps; ++c)
				W[r*y_maps + c] = dCdW_exp[r*y_maps + c] - max_w;
		}		
	}
	

}

// Update importance weights
/* LaTex equation
\pi _{i}^{(m)} = \frac {exp^{-w_{i}^{(m)}}} {\sum_{m'} exp^{-w_{i}^{(m')}}} 
*/
void MMTSNE::update_imp_W(const std::vector<double> &W) {	
	for (size_t ri = 0; ri < x_rows; ++ri) {
		double sum = DBL_MIN;
		for (size_t m = 0; m < y_maps; ++m) {
			iW[ri*y_maps + m] = exp(W[ri*y_maps + m]);
			sum += iW[ri*y_maps + m];			
		}
		
		for (size_t m = 0; m < y_maps; ++m) {
			iW[ri*y_maps + m] /= sum;			
		}		
	}
}

// Compute the squared Euclidean distance matrix
void MMTSNE::compute_distance(const std::vector<double> &M, const size_t &dim, 
	std::vector<double> &DD) {
	//DD.assign(x_rows * x_rows, 0);
	for (size_t ri = 0; ri < x_rows; ++ri) {
		DD[ri*x_rows + ri] = 0;
		for (size_t rj = 0; rj < ri; ++rj) {
			for (size_t d = 0; d < dim; ++d) {
				DD[ri*x_rows + rj] += pow(M[ri*x_rows + d] - M[rj*x_rows + d], 2);
			}
			DD[rj*x_rows + ri] = DD[ri*x_rows + rj];
		}
	}
}	

// Compute input similarities using a Gaussian kernel with a fixed perplexity
void MMTSNE::compute_Gaussian_kernel(const double *X_dist, double *P,
	size_t row_from, size_t row_to, size_t thread_id) {
	// Compute Gaussian kernel row by row
	clock_t start = clock();
	for (size_t r = row_from; r < row_to; ++r) {
		// Initialize some variables		
		double beta = 1.0;
		double min_beta = -DBL_MAX, max_beta = DBL_MAX;
		double tol = 1e-5;
		double sum_P = DBL_MIN;

		// Iterate until a good perplexity is found using Binary Search
		for (size_t iter = 0; iter < 200; ++iter) {
			double H = 0;
			// Compute Gaussian kernel row
			for (size_t c = 0; c < x_rows; ++c) {
				if (r == c) P[r*x_rows + c] = DBL_MIN;
				else P[r*x_rows + c] = exp(-beta * X_dist[r*x_rows + c]);
				H += beta * (X_dist[r*x_rows + c] * P[r*x_rows + c]);
			}

			// Compute entropy of current row
			for (size_t c = 0; c < x_rows; ++c) {
				sum_P += P[r*x_rows + c];
			}			
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
		for (size_t c = 0; c < x_rows; ++c) P[r*x_rows + c] /= sum_P;
	}
	clock_t end = clock();
	if (verbose) {
		std::cout << "\t\tSNE | Thread #" << thread_id << " has ended | Time taken: " <<
			std::setprecision(3) << (end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
	}
}

// Stochastic Neighborhood Embedding: convert high-dimensional Euclidean distance
// between data points into conditional probabilities that represent similarities
void MMTSNE::compute_SNE(std::vector<double> &P) {
	std::vector<std::thread> thread_pool;
	size_t max_threads = std::thread::hardware_concurrency() + 1;
	size_t block = x_rows / max_threads;

	std::vector<double> X_dist(x_rows * x_rows, 0);
	// Compute the squared Euclidean distance matrix
	compute_distance(X, x_dims, X_dist);

	if (block == 0) {
		// Since (rows < max_threads) do not launch threads
		compute_Gaussian_kernel(X_dist.data(), P.data(), 0, x_rows, 0);
	}
	else {
		for (size_t r = 0, thread_id = 1; r < x_rows; r += block, ++thread_id) {			
			if (verbose) {
				std::cout << "\t\tSNE | Launching thread #" << thread_id << " | Rows [" << r <<
					", " << (((r + block) > x_rows)? x_rows : (r + block)) << "]" << 
					std::endl;
			}
			thread_pool.push_back(std::thread(&MMTSNE::compute_Gaussian_kernel, this,
					X_dist.data(), P.data(), r, (((r + block) > x_rows) ? x_rows : (r + block)), 
					thread_id));
		}
	}
	// Synchronize threads
	for (auto &t : thread_pool) {
		if (t.joinable()) t.join();
	}
		
	// Symmetrize P
	double sum = std::accumulate(P.begin(), P.end(), DBL_MIN);
	for (size_t ri = 0; ri < x_rows; ++ri) {
		P[ri*x_rows + ri] = (P[ri*x_rows + ri] * 2) / sum;		
		for (size_t rj = 0; rj < ri; ++rj) {
			P[ri*x_rows + rj] = (P[ri*x_rows + rj] + P[rj*x_rows + ri]) / sum;
			P[rj*x_rows + ri] = P[ri*x_rows + rj];			
		}
	}
}

// Normalizes matrix (zero mean in the range [-1,1]
void MMTSNE::normalize(std::vector<double> &M, const size_t &rows, const size_t &cols) {
	// Matrix mean
	double sum = std::accumulate(M.begin(), M.end(), 0.0);
	double mean = sum / M.size();
	
	double abs_max = DBL_MIN;
	double abs_min = DBL_MAX;

	for (size_t i = 0; i < M.size(); ++i) {		
		if (M[i] > abs_max) abs_max = M[i];
		if (M[i] < abs_min) abs_min = M[i];
	}

	// Mean subtracted absolute max element in matrix
	abs_max = abs(abs_max - mean);
	abs_min = abs(abs_min - mean);
	double max = abs_max > abs_min ? abs_max : abs_min;
	
	// Normalize matrix
	for (size_t r = 0; r < rows; ++r) {
		for (size_t c = 0; c < cols; ++c) {
			M[r*cols + c] = (M[r*cols + c] - mean) / (max + DBL_MIN);
		}
	}	
}


// Load high-dimensional input vector data from a CSV file (default perplexity: 30)
bool MMTSNE::load_input_vectors_csv(const std::string &fileName, const char &delimiter) {
	return load_input_vectors_csv(fileName, delimiter, 30);
}

// Load high-dimensional input vector data from a CSV file with perplexity value
bool MMTSNE::load_input_vectors_csv(const std::string &fileName, const char &delimiter,
	const size_t &perplexity) {
	std::ifstream input_csv;

	std::string line;
	size_t rows = 0;
	size_t cols_prev = 0, cols = 0;
	clock_t start, end;

	std::cout << "Loading input vectors matrix..." << std::endl;

	start = clock();
	try {
		input_csv.open(fileName);
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
	}
	catch (std::exception &e) {
		std::cout << "Exception loading input vector file: " << e.what() << std::endl;
		return false;
	}

	end = clock();

	// Set class variables
	x_rows = rows;
	x_dims = cols;
	this->perplexity = perplexity;
	status = input_vectors;

	std::cout << "\t Done. Matrix of size: " << rows << " x " << cols
		<< " loaded." << "\n\t Time taken: " << std::setprecision(5) <<
		(double)(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;

	return true;
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

	// Normalize P matrix
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

	return true;
}

// Save low-dimensional output data to a CSV file
void MMTSNE::save_output_csv(const std::string &fileName) {
	std::ofstream output_csv;
	try {
		// Save low-dimensional vectors Y & importance weights iW to file
		output_csv.open(fileName + ".csv", std::ios::out);
		for (size_t m = 0; m < y_maps; ++m) {
			size_t base1 = m * x_rows * y_dims;
			for (size_t r = 0; r < x_rows; ++r) {
				size_t base2 = r * y_dims;
				output_csv << (m + 1) << "," << iW_best[r*y_maps + m] << ",";
				for (size_t d = 0; d < y_dims; ++d) {
					output_csv << Y_best[base1 + base2 + d] << (d == (y_dims - 1) ? "\n" : ",");
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

	// Save map statistics to file
	try {		
		output_csv.open(fileName + "_stats.txt", std::ios::out);
		for (size_t m = 0; m < y_maps; ++m) {
			double sum_w = 0;
			int ws[10] = { 0 };
			for (size_t r = 0; r < x_rows; ++r) {
				sum_w += iW[r*y_maps + m];
				if (iW[r*y_maps + m] > 0.9) ws[9] += 1;
				else if (iW[r*y_maps + m] > 0.8) ws[8] += 1;
				else if (iW[r*y_maps + m] > 0.7) ws[7] += 1;
				else if (iW[r*y_maps + m] > 0.6) ws[6] += 1;
				else if (iW[r*y_maps + m] > 0.5) ws[5] += 1;
				else if (iW[r*y_maps + m] > 0.4) ws[4] += 1;
				else if (iW[r*y_maps + m] > 0.3) ws[3] += 1;
				else if (iW[r*y_maps + m] > 0.2) ws[2] += 1;
				else if (iW[r*y_maps + m] > 0.1) ws[1] += 1;
				else ws[0] += 1;
			}
			std::cout << "Map # " << m << " | Sum of weights: " << sum_w
				<< std::endl;
			output_csv << "Map # " << m << " | Sum of weights: " << 
				std::setprecision(4) << sum_w << std::endl;
			for (size_t i = 0; i < 10; ++i) {
				output_csv << "  " << std::setprecision(3) << (float)(i) / 10
					<< " - " << (float)(i + 1) / 10 << " : " << ws[i] << std::endl;
			}
		}
		output_csv.close();
		std::cout << "Output map statistics files saved" << std::endl;
	}
	catch (std::exception &e) {
		std::cout << "Exception while saving map statistics output file: " 
			<< e.what() << std::endl;
		if (output_csv.is_open()) output_csv.close();		
	}

	// Save error list to file
	try {		
		output_csv.open(fileName + "_KL_errors.csv", std::ios::out);
		for (auto &err : error_list) {
			output_csv << std::setprecision(15) << err << "\n";
		}
		output_csv.close();
		std::cout << "Output KL errors saved to file" << std::endl;
	}
	catch (std::exception &e) {
		std::cout << "Exception while saving KL errors output file: " 
			<< e.what() << std::endl;
		if (output_csv.is_open()) output_csv.close();		
	}	
}
