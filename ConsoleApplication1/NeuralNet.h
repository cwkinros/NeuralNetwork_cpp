#pragma once
#include <armadillo>
#include "layer.h"
#include <fstream>
#include <iostream>

using namespace std;
using namespace arma;

class NeuralNet {
public:
	NeuralNet();
	NeuralNet(int n_i, int n_o);
	NeuralNet(int n_i, int n_o, int n_l);
	NeuralNet(int n_i, int n_o, int n_l, vec ns);
	NeuralNet(int n_i, int n_o, int n_l, vec ns, int nlin);
	NeuralNet(int n_i, int n_o, int n_l, vec ns, int nlin, mat ws[]);
	NeuralNet(int n_i, int n_o, int n_l, vec ns, mat ws[]);

	void train_GD(mat input, mat output, int n_steps, float lr, bool print, string filename);
	void train_GD_Alr(mat input, mat output, int n_steps, float ilr, float inc, float dec, bool print, string filename);
	void train_SGD(mat input, mat output, int n_steps, float a, float b, bool print, string filename);
	float test(mat input, mat output);
	float accuracy_test(mat input, vec output);
	void print_weights();
	void print_grad();
	mat apply(mat input);
	void forwback(mat input, mat output);
	vec Hv(vec v);

private:
	void initialize_params(int n_i, int n_o, int n_l);
	void initialize_layers(vec ns, mat ws[], int nlin);
	void initialize_layers(int n);
	void initialize_layers(vec ns);
	void initialize_layers(vec ns, int nlin);
	void initialize_layers(vec ns, mat ws[]);
	mat forward_prop(mat input);
	void back_prop(mat dz);
	void step(float lr);
	float error;
	int output_size;
	int input_size;
	int m;
	int n_layers;
	Layer* Layers;
};
