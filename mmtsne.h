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

typedef std::vector<std::unique_ptr<Eigen::MatrixXd>> Eigen_Matrix3Xd;		

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
	Eigen::MatrixXd X;					// Normalized input high-dimensional dataset
	Eigen_Matrix3Xd Y;					// Output low-dimensional dataset								
	Eigen::MatrixXd iW;					// Importance weights	
	Eigen::MatrixXd P = Eigen::MatrixXd::Zero(X.rows(), X.rows());			// Input pairwise similarity matrix
	
	// MM t-SNE parameters
	int y_dims;							// Number of dimensions in low-dimensional dataset Y
	int y_maps;							// Number of maps in MM t-SNE	
	double perplexity = 30; 			// A smooth measure of the effective number of neighbours: Perp(P_{i}) = 2^{H(P_{i})} where H(P_{i}) is Shannon entropy of P_{i}

	State status = empty;

	bool verbose = false;

	// Evaluate cost function as Kullback-Liebler divergence between P & Q
	double compute_KL(const Eigen::MatrixXd &P, const Eigen::MatrixXd &Q);

	// Compute output pairwise similarity matrix Q
	void compute_similarity_Q(Eigen::MatrixXd &Q, double &Z, 
		const Eigen_Matrix3Xd &YD);

	// Compute gradient of cost function w.r.t low dimensional map points
	void Y_gradients(Eigen_Matrix3Xd &dCdY, const Eigen::MatrixXd &P,
		const Eigen::MatrixXd &Q, const Eigen_Matrix3Xd &YD, const double &Z);

	// Compute gradient of cost function w.r.t unconstrained weights
	void W_gradients(Eigen::MatrixXd &dCdW, const Eigen::MatrixXd &P,
		const Eigen::MatrixXd &Q, const Eigen_Matrix3Xd &YD, const double &Z);
	
	// Compute importance weights expressed in terms of unconstrained weights
	void update_imp_W(const Eigen::MatrixXd &W);


	// Compute input similarities using a Gaussian kernel with a fixed perplexity
	void compute_Gaussian_kernel(const Eigen::MatrixXd &X_dist, Eigen::MatrixXd &P,
		size_t row_from, size_t row_to, size_t thread_id);

	// Compute the squared Euclidean distance matrix
	void compute_distance(const Eigen::MatrixXd &M, Eigen::MatrixXd &DD);

	// Stochastic Neighborhood Embedding
	void compute_SNE(Eigen::MatrixXd &P);

	// Normalizes matrix (zero mean in the range [-1,1]
	void normalize(Eigen::MatrixXd &M);

	// Symmetrize matrix
	void symmetrize(Eigen::MatrixXd &M);

	// Create 3D Eigen matrix initialized with 'value'
	Eigen_Matrix3Xd Matrix3Xd(const size_t& rows, const size_t& columns,
		const size_t& depth, const double& value);

};