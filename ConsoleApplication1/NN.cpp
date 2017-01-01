#include "NN.h"
#include <iostream>

NeuralNetwork::NeuralNetwork() {
	head_set = false;
	tail_set = false;
	this->initialize_params(2, 1, 2);
	this->initialize_layers(10);
}

NeuralNetwork::NeuralNetwork(int n_i, int n_o) {
	head_set = false;
	tail_set = false;
	this->initialize_params(n_i, n_o, 2);
	this->initialize_layers(10);
}

NeuralNetwork::NeuralNetwork(int n_i, int n_o, int n_l) {
	head_set = false;
	tail_set = false;
	this->initialize_params(n_i, n_o, n_l);
	this->initialize_layers(10);
}

NeuralNetwork::NeuralNetwork(Layer h) {
	input_size = h.get_input_n();
	
	Layer current = h;
	head = &h;
	int count = 1;
	while (current.get_next().get_output_n() != 0) {
		current = current.get_next();
		count++;
	}
	n_layers = count;
	tail = &current;
	output_size = current.get_output_n();

}

void NeuralNetwork::initialize_params(int n_i, int n_o, int n_l) {
	input_size = n_i;
	output_size = n_o;
	n_layers = n_l;
}

void NeuralNetwork::initialize_layers(int n) {
	if (n_layers == 1) {
		Layer head(output_size, 0, input_size);
		Layer* tail = &head;
	}
	else {
		Layer last(n, 0, input_size);
		head = &last;
		Layer* next = nullptr;
		Layer* previous = nullptr;
		for (int i = 1; i < n_layers - 1; i++) {
			Layer new_node(n, 0,  n);
			next = &new_node;
			last.set_next(next);
			previous = &last;
			new_node.set_previous(previous);
			last = new_node;
		}
		Layer tail_2b(output_size, 0, n);
		tail = &tail_2b;
		next = &tail_2b;
		last.set_next(next);
		previous = &last;
		tail->set_previous(previous);
	}
	head_set = true;
	tail_set = true;
}

Matrix NeuralNetwork::forward_prop(Matrix layer_input) {
	Layer* layer = this->head;
	Matrix result = layer_input;
	while (layer->get_output_n() != 0) {
		result = layer->output(result);
		layer = &(layer->get_next());
	}
	return result;
}

void NeuralNetwork::backward_prop(Matrix errors) {
	Layer current = *tail;
	Matrix* last_errors = &errors;
	while (current.get_output_n() != 0) {
		last_errors = &(current.back_prop(*(last_errors)));
		current = current.get_previous();
	}
}

void NeuralNetwork::step(float learning_rate) {
	Layer current = *head;
	while (current.get_output_n() != 0) {
		current.step(learning_rate);
	}
}

void NeuralNetwork::train_GD(Matrix &input, Matrix &expected_output) {
	Matrix* out;
	Matrix* errors;
	float lr = 0.01f;
	for (int i = 0; i < 200; i++) {
		out = this->forward_prop(input);
		errors = expected_output.sub(*out);
		this->backward_prop(*errors);
		this->step(lr);
		std::cout << "error: " << this->test(input, expected_output) << std::endl;
	}
}

float NeuralNetwork::test(Matrix input, Matrix expected_output) {
	Matrix out = this->apply(input);
	Matrix errors = expected_output.sub(out);
	return errors.sum_squared();
}

Matrix NeuralNetwork::apply(Matrix input) {
	return this->forward_prop(input);
}
