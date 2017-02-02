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
	ballSize = 1000;
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
	m = input.n_cols;
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

void NeuralNet::forwback(mat input, mat expected_output) {
	mat out = forward_prop(input);
	mat errors = out - expected_output;
	back_prop(errors);
}

mat NeuralNet::apply(mat input) {
	return forward_prop(input);
}

void NeuralNet::print_weights() {
	for (int i = 0; i < n_layers; i++) {
		Layers[i].print_weights();
	}
}


void NeuralNet::train_TRM(mat input, mat expected_output, int n_steps, float bs, bool print, string filename) {
	vec out;
	vec errors;
	ballSize = bs;
	ofstream file;
	bool log;
	float n1 = 0.10, n2 = 0.25, n3 = 0.25;
	if (filename == "") {
		log = false;
	}
	else {
		file.open(filename, ios::out);
		log = true;
	}
	int next_error;
	out = forward_prop(input);
	errors = out - expected_output;
	back_prop(errors);
	error = float(accu((errors) % (errors)));
	for (int i = 0; i < n_steps; i++) {
		while (true) {
			step_TRM();
			out = forward_prop(input);
			errors = out - expected_output;
			back_prop(errors);
			next_error = float(accu((errors) % (errors)));

		}
		if (print) {
			error = float(accu(abs(errors)));
			std::cout << "error: " << error << std::endl;
			if (log) {
				file << error << endl;
			}
		}

		error = next_error;
	}
}

void NeuralNet::step_TRM() {
	// it's all about getting info into little vectors and stuff.... oh goodness
	// vecs can keep expanding i think - look it up

	int size = 0;
	for (int i = 0; i < n_layers; i++) {
		size += Layers[i].W.n_elem;
	}

	vec g(size);
	vec eigvec(2*size);

	int count = 0;
	int l = 0;
	while (l < n_layers) {
		for (int j = 0; j < Layers[l].W.n_elem; j++) {
			g(count) = Layers[l].GradW(j);
			count++;
		}
		l++;
	}
	float eigvalue = power_series(eigvec, g);
	vec Y1 = eigvec.subvec(0, size - 1);
	vec Y2 = eigvec.subvec(size, size * 2 - 1);
	int sign = 1;
	if (dot(g,Y2) < 0) {
		sign = -1;
	}
	vec p_star = -sign*ballSize*Y1 / accu(abs(Y1));
	
	l = 0;
	int n;
	int current = 0;
	while (l < n_layers) {
		n = Layers[l].W.n_elem;
		Layers[l].step_TRM(p_star.subvec(current, current + n - 1));
		current += n;
	}

}

float NeuralNet::power_series(vec &eigvec, vec g) {
	vec w(eigvec.n_elem);
	float eigvalue;
	w = Mv(eigvec, g);
	for (int k = 0; k < eigvec.n_elem; k++) {
		eigvec = w / accu(abs(w));
		w = Mv(eigvec, g);
		eigvalue = dot(eigvec,w);
	}
	return eigvalue;
}

vec NeuralNet::Mv(vec v, vec g) {
	vec y1(v.n_elem / 2);
	vec y2(v.n_elem / 2);
	y1 = v.subvec(0,(v.n_elem / 2)-1);
	y2 = v.subvec(v.n_elem / 2, v.n_elem - 1);
	vec hv(v.n_elem);
	hv.subvec(0, (v.n_elem / 2) - 1) = -Hv(y1) + g*(g.t()*y2)/ballSize;
	hv.subvec((v.n_elem / 2), v.n_elem - 1) = y1 - Hv(y2);
	return hv;
}

vec NeuralNet::Hv(vec v) {
	/*
		I use col by col matrix to vec and layer by layer

		example: V1 = [1,2,3;4,5,6] and V2 = [7,8;9,10]
		v = [1,4,2,5,3,6,7,9,8,10]

		let l be the starting index for layer l values

		v[l + r + c*n_rows] = Vl(r,c);
		and vice versa	
		
	*/

	vec hv(v.n_elem);
	Layer* layer = &Layers[0];
	mat result(layer->get_input_n(),m);
	result.fill(0.0f);
	int rows, cols;
	int l = 0;
	int* ls = new int[n_layers];

	int c = 0;

	mat* vs = new mat[n_layers];
	while (layer) {
		rows = layer->W.n_rows;
		cols = layer->W.n_cols;
		vs[c] = mat(rows,cols);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				vs[c](i, j) = v[l + i + j*rows];
			}
		}
		
		result = layer->forwardHv(result,vs[c]);
		layer = layer->get_next();
		ls[c] = l;
		l += rows*cols;
		c ++;
	}

	layer = &Layers[n_layers - 1];
	c -= 1;

	while (layer) {
		result = layer->backHv(result, vs[c]);
		l = ls[c];
		rows = layer->W.n_rows;
		cols = layer->W.n_cols;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				hv(l + i + j*rows) = layer->R_dw(i,j);
			}
		}
		layer = layer->get_previous();
		c--;
	}

	return hv;
}

void NeuralNet::print_grad() {
	for (int i = 0; i < n_layers; i++) {
		Layers[i].print_grad();
	}
}

