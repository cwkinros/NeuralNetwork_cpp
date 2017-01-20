#include "NeuralNet.h"


NeuralNet::NeuralNet() {
	initialize_params(2, 1, 2);
	initialize_layers(10);
}

NeuralNet::NeuralNet(int n_i, int n_o) {
	initialize_params(n_i, n_o, 2);
	initialize_layers(10);
}

NeuralNet::NeuralNet(int n_i, int n_o, int n_l) {
	initialize_params(n_i, n_o, n_l);
	initialize_layers(10);
}

NeuralNet::NeuralNet(int n_i, int n_o, int n_l, vec ns) {
	initialize_params(n_i, n_o, n_l);
	initialize_layers(ns);
}

NeuralNet::NeuralNet(int n_i, int n_o, int n_l, vec ns, int nlin) {
	initialize_params(n_i, n_o, n_l);
	initialize_layers(ns, nlin);
}

NeuralNet::NeuralNet(int n_i, int n_o, int n_l, vec ns, int nlin, mat ws[]) {
	initialize_params(n_i, n_o, n_l);
	initialize_layers(ns, ws, nlin);

}
NeuralNet::NeuralNet(int n_i, int n_o, int n_l, vec ns, mat ws[]) {
	initialize_params(n_i, n_o, n_l);
	initialize_layers(ns, ws);
}

void NeuralNet::initialize_params(int n_i, int n_o, int n_l) {
	input_size = n_i;
	output_size = n_o;
	n_layers = n_l;
	Layers = new Layer[n_l];
}

void NeuralNet::initialize_layers(int n) {
	if (n_layers == 1) {
		Layers[0] = Layer(output_size, 0, input_size);
	}
	else {
		Layers[0] = Layer(n, 0, input_size);
		Layer* last = &Layers[0];
		Layer* next;
		Layer* previous;
		Layer new_node;
		for (int i = 1; i < n_layers - 1; i++) {
			Layers[i] = Layer(n, 0, n);
			next = &Layers[i];
			last->set_next(next);
			previous = last;
			Layers[i].set_previous(previous);
			last = &Layers[i];
		}
		Layers[n_layers - 1] = Layer(output_size, 0, n);
		last->set_next(&Layers[n_layers - 1]);
		Layers[n_layers - 1].set_previous(last);
	}
}

void NeuralNet::initialize_layers(vec ns) {
	if (n_layers == 1) {
		Layers[0] = Layer(output_size, 0, input_size);
	}
	else {
		Layers[0] = Layer(ns[0], 0, input_size);
		Layer* last = &Layers[0];
		Layer* next;
		Layer* previous;
		Layer new_node;
		for (int i = 1; i < n_layers - 1; i++) {
			Layers[i] = Layer(ns[i], 0, ns[i-1]);
			next = &Layers[i];
			last->set_next(next);
			previous = last;
			Layers[i].set_previous(previous);
			last = &Layers[i];
		}
		Layers[n_layers - 1] = Layer(output_size, 0, ns[n_layers - 2]);
		last->set_next(&Layers[n_layers - 1]);
		Layers[n_layers - 1].set_previous(last);
	}
}

void NeuralNet::initialize_layers(vec ns, int nlin) {
	if (n_layers == 1) {
		Layers[0] = Layer(output_size, nlin, input_size);
	}
	else {
		Layers[0] = Layer(ns[0], 0, input_size);
		Layer* last = &Layers[0];
		Layer* next;
		Layer* previous;
		Layer new_node;
		for (int i = 1; i < n_layers - 1; i++) {
			Layers[i] = Layer(ns[i], 0, ns[i - 1]);
			next = &Layers[i];
			last->set_next(next);
			previous = last;
			Layers[i].set_previous(previous);
			last = &Layers[i];
		}
		Layers[n_layers - 1] = Layer(output_size, nlin, ns[n_layers - 2]);
		last->set_next(&Layers[n_layers - 1]);
		Layers[n_layers - 1].set_previous(last);
	}
}

void NeuralNet::initialize_layers(vec ns, mat ws[]) {
	if (n_layers == 1) {
		Layers[0] = Layer(output_size, 0, input_size, ws[0]);
	}
	else {
		Layers[0] = Layer(ns[0], 0, input_size, ws[0]);
		Layer* last = &Layers[0];
		Layer* next;
		Layer* previous;
		Layer new_node;
		for (int i = 1; i < n_layers - 1; i++) {
			Layers[i] = Layer(ns[i], 0, ns[i-1], ws[i]);
			next = &Layers[i];
			last->set_next(next);
			previous = last;
			Layers[i].set_previous(previous);
			last = &Layers[i];
		}
		Layers[n_layers - 1] = Layer(output_size, 0, ns[n_layers-2], ws[n_layers - 1]);
		last->set_next(&Layers[n_layers - 1]);
		Layers[n_layers - 1].set_previous(last);
	}
}

void NeuralNet::initialize_layers(vec ns, mat ws[], int nlin) {
	if (n_layers == 1) {
		Layers[0] = Layer(output_size, nlin, input_size, ws[0]);
	}
	else {
		Layers[0] = Layer(ns[0], 0, input_size, ws[0]);
		Layer* last = &Layers[0];
		Layer* next;
		Layer* previous;
		Layer new_node;
		for (int i = 1; i < n_layers - 1; i++) {
			Layers[i] = Layer(ns[i], 0, ns[i - 1], ws[i]);
			next = &Layers[i];
			last->set_next(next);
			previous = last;
			Layers[i].set_previous(previous);
			last = &Layers[i];
		}
		Layers[n_layers - 1] = Layer(output_size, nlin, ns[n_layers - 2], ws[n_layers - 1]);
		last->set_next(&Layers[n_layers - 1]);
		Layers[n_layers - 1].set_previous(last);
	}
}


mat NeuralNet::forward_prop(mat input) {
	Layer* layer = &Layers[0];
	mat result = input;
	while (layer){
		result = layer->output(result);
		layer = layer->get_next();
	}
	return result;
}

void NeuralNet::back_prop(mat dz) {
	Layer* current;
	if (n_layers == 1) {
		current = &Layers[0];
	}
	else {
		current = &Layers[n_layers - 1];
	}
	mat last_dz = dz;
	//dz.print("2*error: ");
	while (current && current->get_output_n() != 0) {
		last_dz = (current->back_prop(last_dz));
		current = current->get_previous();
	}
}

void NeuralNet::step(float lr) {
	Layer* current = &Layers[0];
	while (current) {
		current->step(lr);
		current = current->get_next();
	}
}


void NeuralNet::train_GD(mat input, mat expected_output, int n_steps, float lr, bool print, string filename) {
	bool log;
	ofstream file;
	if (filename == "") { 
		log = false; 
	}
	else {
		file.open(filename, ios::out);
		log = true;
	}
	mat out;
	mat errors;
	float error;
	float last_error = INFINITY;
	for (int i = 0; i < n_steps; i++) {
		out = forward_prop(input);
		errors = out-expected_output;
		
		back_prop(errors);
		step(lr);
		if (print) {
			error = float(accu(abs(errors)));
			std::cout << "error: " << error << std::endl;
			if (log) {
				file << error << endl;
			}
		}

	}
	
}

void NeuralNet::train_SGD(mat input, mat expected_output, int n_steps, float a, float b, bool print, string filename) {
	bool log;
	ofstream file;
	if (filename == "") {
		log = false;
	}
	else {
		file.open(filename, ios::out);
		log = true;
	}
	mat out;
	mat errors;
	float error;
	float last_error = INFINITY;
	float lr = a / b;
	int m = input.n_cols;
	int c;
	for (int i = 0; i < n_steps; i++) {
		out = forward_prop(input);
		errors = out - expected_output;
		c = rand() % m;
		back_prop(errors.col(c));
		step(lr);
		if (print) {
			error = float(accu(abs(errors)));
			std::cout << "error: " << error << std::endl;
			if (log) {
				file << error << endl;
			}
		}
		lr = a / (b + i);

	}
}

void NeuralNet::train_GD_Alr(mat input, mat expected_output, int n_steps, float ilr, float inc, float dec, bool print, string filename) {
	ofstream file;
	bool log;
	if (filename == "") {
		log = false;
	}
	else {
		file.open(filename, ios::out);
		log = true;
	}
	mat out;
	mat errors;
	float error;
	float last_error = INFINITY;
	float lr = ilr;
	for (int i = 0; i < n_steps; i++) {
		out = forward_prop(input);
		errors = out - expected_output;

		back_prop(errors);
		step(lr);
		if (print) {
			error = float(accu(abs(errors)));
			std::cout << "error: " << error << std::endl;
			if (log) {
				file << error << endl;
			}
		}

		error = float(accu(abs(errors)));
		if (last_error < error) {
		lr = lr * dec;
		}
		else {
		lr = lr* inc;
		}
		last_error = error;
	}
}
 
float NeuralNet::test(mat input, mat output) {
	mat out = forward_prop(input);
	mat errors = output - out;
	return float(accu(abs(errors)));
}

float NeuralNet::accuracy_test(mat input, vec labels) {
	mat outputs = apply(input);
	int idx, correct = 0;
	for (unsigned int i = 0; i < outputs.n_cols; i++) {
		idx = outputs.col(i).index_max();
		if (idx == labels(i)) {
			correct++;
		}
	}
	return float(correct)*100.0f / float(input.n_cols) ;
}

mat NeuralNet::apply(mat input) {
	return forward_prop(input);
}

void NeuralNet::print_weights() {
	for (int i = 0; i < n_layers; i++) {
		Layers[i].print_weights();
	}
}

void NeuralNet::print_grad() {
	for (int i = 0; i < n_layers; i++) {
		Layers[i].print_grad();
	}
}

