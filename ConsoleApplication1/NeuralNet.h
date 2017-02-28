#pragma once
#include <armadillo>
#include "layer.h"
#include <fstream>
#include <iostream>
#include <time.h>

using namespace std;
using namespace arma;

class NeuralNet {
public:
	NeuralNet();
	NeuralNet(int n_i, int n_o);
	NeuralNet(int n_i, int n_o, int n_l);
	NeuralNet(int n_i, int n_o, int n_l, vec ns);
	NeuralNet(int n_i, int n_o, int n_l, int nlin);
	NeuralNet(int n_i, int n_o, int n_l, vec ns, int nlin);
	NeuralNet(int n_i, int n_o, int n_l, vec ns, int nlin, mat ws[]);
	NeuralNet(int n_i, int n_o, int n_l, vec ns, vec nlin);
	NeuralNet::NeuralNet(int n_i, int n_o, int n_l, vec ns, mat ws[]);
	void train_TRM(mat input, mat output, int n_steps, float ballSize, bool print, string filename, float cutoff=0.0f, bool times = false, int max_time=5*1000*60);
	void train_TRM_cd(mat input, mat output, int n_steps, float ballSize, bool print, string filename, float cutoff = 0.0f, bool times = false, int max_time = 5 * 1000 * 60);
	void train_GD(mat input, mat output, int n_steps, float lr, bool print, string filename, bool times = false, int max_time = 5*1000*60);
	void train_GD_Alr(mat input, mat output, int n_steps, float ilr, float inc, float dec, bool print, string filename);
	void train_SGD(mat input, mat output, int n_steps, float a, float b, bool print, string filename);
	float test(mat input, mat output);
	float accuracy_test(mat input, vec output);
	void print_weights();
	void print_grad();
	mat apply(mat input);
	void forwback(mat input, mat output);
	
	vec Hv(vec v);
	vec M_squiggle_v(vec v, vec g,float lambda);
	vec Mv(vec v, vec g);
	vec cg(vec g);
	vec get_weights();
	void set_weights(vec weights);
	float power_series(vec &eigvec, vec g);
	void get_p1_TRM(vec &p_star, vec &g);
	void set_TRM_parameters(float lowerbound, float upperbound, float shrink, float grow);
private:
	void add_p(vec p_star);
	int size;
	void initialize_params(int n_i, int n_o, int n_l);
	void initialize_layers(vec ns, mat ws[], int nlin);
	void initialize_layers(int n);
	void initialize_layers(int n, int nlin);
	void initialize_layers(vec ns);
	void initialize_layers(vec ns, int nlin);
	void initialize_layers(vec ns, mat ws[]);
	void initialize_layers(vec ns, vec nlin);
	mat forward_prop(mat input);
	void back_prop(mat dz);
	void step(float lr);
	double error;
	float ballSize;
	int output_size;
	int input_size;
	int m;
	int num_weights;
	int n_layers;
	Layer* Layers;
	float lb, ub, shrink, grow;
};
