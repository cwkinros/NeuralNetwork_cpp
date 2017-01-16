#pragma once
#include <armadillo>
#include "layer.h"
using namespace arma;

class NeuralNet {
public:
	NeuralNet();
	NeuralNet(int n_i, int n_o);
	NeuralNet(int n_i, int n_o, int n_l);
	NeuralNet(int n_i, int n_o, int n_l, vec ns);
	NeuralNet(int n_i, int n_o, int n_l, vec ns, int nlin);
	NeuralNet(int n_i, int n_o, int n_l, vec ns, mat ws[]);

	void train_GD(mat input, mat output, int n_steps, float lr, bool print);
	float test(mat input, mat output);
	float accuracy_test(mat input, vec output);
	void print_weights();
	void print_grad();
	mat apply(mat input);

private:
	void initialize_params(int n_i, int n_o, int n_l);
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
	int n_layers;
	Layer* Layers;
};
