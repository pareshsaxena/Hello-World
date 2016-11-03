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

#pragma once
#include <memory>
#include <vector>
#include <Eigen/Dense>

enum State { input_vectors, input_probability, empty };

class MMTSNE
{
public:
	// Class constructor
	MMTSNE();
	MMTSNE(double perplexity);

	// Class destructor
	~MMTSNE();

	// Perform MM t-SNE
	void construct_maps();
	void construct_maps(bool verbose);
	void construct_maps(int y_dims, int y_maps, int iterations, bool verbose);

	// Load high-dimensional input vector data from a CSV file
	bool load_input_vectors_csv(const std::string &fileName, const char &delimiter);
	// Load high-dimensional input probability data from a CSV file
	bool load_input_probability_csv(const std::string &fileName, const char &delimiter);
	// Save low-dimensional output vector data to a CSV file
	bool save_output_vectors_csv(const std::string &fileName, const char &delimiter);

	
private:
	// MM t-SNE data structures
	std::vector<double> X;							// Normalized input high-dimensional dataset
	std::vector<double> Y;							// Output low-dimensional dataset								
	std::vector<double> iW;							// Importance weights	
	std::vector<double> P;							// Input pairwise similarity matrix
	
	// MM t-SNE variables
	size_t x_rows;									// Number of vectors in high-dimensional dataset X
	size_t x_dims;									// Number of dimensions in low - dimensional dataset Y	
	size_t y_dims;									// Number of dimensions in low-dimensional dataset Y
	size_t y_maps;									// Number of maps in MM t-SNE	
	double perplexity = 30; 						// A smooth measure of the effective number of neighbours: Perp(P_{i}) = 2^{H(P_{i})} where H(P_{i}) is Shannon entropy of P_{i}
	
	State status = empty;

	bool verbose = false;

	// Evaluate cost function as Kullback-Liebler divergence between P & Q
	double compute_KL(const std::vector<double> &P, const std::vector<double> &Q);

	// Compute output pairwise similarity matrix Q
	void compute_similarity_Q(std::vector<double> &Q, double &Z, 
		const std::vector<double> &YD);

	// Compute gradient of cost function w.r.t low dimensional map points
	void Y_gradients(double *dCdY, double *dCdY_exp, double *dCdD, double *epsilon_Y,
		const std::vector<double> &P, const std::vector<double> &Q, const std::vector<double> &YD,
		const double &Z, const double &alpha, const double &epsilon_inc, const double &epsilon_dec);

	// Compute gradient of cost function w.r.t unconstrained weights
	void W_gradients(double *dCdW, double *dCdW_exp, double *dCdP, double *W,
		double *epsilon_W, const std::vector<double> &P, const std::vector<double> &Q,
		const std::vector<double> &YD, const double &Z, const double &alpha,
		const double &epsilon_inc, const double &epsilon_dec);
	
	// Compute importance weights expressed in terms of unconstrained weights
	void update_imp_W(const std::vector<double> &W);


	// Compute input similarities using a Gaussian kernel with a fixed perplexity
	void compute_Gaussian_kernel(const std::vector<double> &X_dist, std::vector<double> &P,
		size_t row_from, size_t row_to, size_t thread_id);

	// Compute the squared Euclidean distance matrix
	void compute_distance(const std::vector<double> &M, const size_t &dim, 
		std::vector<double> &DD);

	// Stochastic Neighborhood Embedding
	void compute_SNE(std::vector<double> &P);

	// Normalizes matrix (zero mean in the range [-1,1]
	void normalize(std::vector<double> &M);

	// Symmetrize matrix
	void symmetrize(std::vector<double> &M);
	
};
