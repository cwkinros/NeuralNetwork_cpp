#include "NN.h"
#include <iostream>

using namespace std;
NeuralNetwork::NeuralNetwork() {
	head_set = false;
	tail_set = false;
	initialize_params(2, 1, 2);
	initialize_layers(10);
}

NeuralNetwork::NeuralNetwork(int n_i, int n_o) {
	head_set = false;
	tail_set = false;
	initialize_params(n_i, n_o, 2);
	initialize_layers(10);
}

NeuralNetwork::NeuralNetwork(int n_i, int n_o, int n_l) {
	head_set = false;
	tail_set = false;
	initialize_params(n_i, n_o, n_l);
	initialize_layers(10);
}


void NeuralNetwork::initialize_params(int n_i, int n_o, int n_l) {
	input_size = n_i;
	output_size = n_o;
	n_layers = n_l;
	Layers = new Layer_fs[n_l];
}


void NeuralNetwork::initialize_layers(int n) {
	if (n_layers == 1) {
		Layers[0] = Layer_fs(output_size, 0, input_size);
	}
	else {
		Layers[0] = Layer_fs(n, 0, input_size);
		Layer_fs* last = &Layers[0];
		Layer_fs* next;
		Layer_fs* previous;
		Layer_fs new_node;
		for (int i = 1; i < n_layers - 1; i++) {
		    Layers[i] = Layer_fs(n, 0,  n);
			next = &Layers[i];
			last->set_next(next);	
			previous = last;
			Layers[i].set_previous(previous);
			last = &Layers[i];
		}
		Layers[n_layers-1] = Layer_fs(output_size, 0, n);
		last->set_next(&Layers[n_layers - 1]);
		Layers[n_layers - 1].set_previous(last);

		Layer_fs last_one = Layers[n_layers - 1];

		previous = NULL;
		next = NULL;
		last = NULL;
		int a = 0;
	}
	cout << "layer dimensions: " << endl;
	for (int i = 0; i < n_layers; i++) {
		cout << "layer: " << i << " input n: " << Layers[i].get_input_n() << " output n: " << Layers[i].get_output_n() << endl;
	}
	int b = 0;
}

Matrix* NeuralNetwork::forward_prop(Matrix* layer_input) {
	Layer_fs* layer = &Layers[0];
	Matrix* result = layer_input;
	while (layer) {
		result = &(layer->output(*result));
		layer = layer->get_next();
	}
	return result;
}

void NeuralNetwork::backward_prop(Matrix* errors) {
	Layer_fs* current;
	if (n_layers == 1) {
		current = &Layers[0];
	}
	else { current = &Layers[n_layers-1]; }
	Matrix* last_errors = errors;
	while (current && current->get_output_n() != 0) {
		last_errors = &(current->back_prop(*(last_errors)));
		current = current->get_previous();
	}
}



void NeuralNetwork::step(float learning_rate) {
	Layer_fs* current = &Layers[0];
	while (current) {
		current->step(learning_rate);
		current = current->get_next();
	}
}

void break_head() {
	int a = 1;
}

float sum_matrix(Matrix* m) {
	float sum = 0.0f;
	for (int i = 0; i < m->get_c(); i++) {
		for (int j = 0; j < m->get_r(); j++) {
			sum += abs(m->get(i, j));
		}
	}
	return sum;
}

void NeuralNetwork::train_GD(Matrix* input, Matrix* expected_output) {
	Matrix* out;
	Matrix* errors;
	float error;
	float mult = 0.0000000000000001f;
	float lr;
	for (int i = 0; i < 100000; i++) {
		lr = mult; // so that you can choose to adjust lr with iteration #
		out = forward_prop(input);
		errors = &(expected_output->sub(*out));
		error = sum_matrix(errors);
		backward_prop(errors);
		step(lr);
		std::cout << "error: " << error << std::endl;
	}
}

float NeuralNetwork::test(Matrix input, Matrix expected_output) {
	Matrix out = apply(input);
	Matrix errors = expected_output.sub(out);
	return (errors).sum_squared();
}

Matrix NeuralNetwork::apply(Matrix input) {
	return input; //this->forward_prop(&input);
}
