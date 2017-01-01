#pragma once
#include "layer.h"


class NeuralNetwork {
public:
	NeuralNetwork();
	NeuralNetwork(int n_i, int n_o); 
	NeuralNetwork(int n_i, int n_o, int n_l);
	NeuralNetwork(Layer r);
	
	void initialize_layers(int n);
	void initialize_params(int n_i, int n_o, int n_l);
	void train_GD(Matrix& input, Matrix& output);
	float test(Matrix input, Matrix output);

	Matrix apply(Matrix input);
	//iterate(float step_size);
private:
	Matrix* forward_prop(Matrix &input);
	void* backward_prop(Matrix &errors);
	void step(float learning_rate);
	float error;
	int output_size;
	int input_size;
	int n_layers;
	Layer* head;
	bool head_set;
	bool tail_set;
	Layer* tail;
};