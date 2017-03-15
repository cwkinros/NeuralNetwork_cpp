#include "NeuralNet.h"


NeuralNet::NeuralNet() {
	initialize_params(2, 1, 2);
	initialize_layers(10);
	size = 0;
	for (int i = 0; i < n_layers; i++) {
		size += Layers[i].W.n_elem;
	}
}

NeuralNet::NeuralNet(int n_i, int n_o) {
	initialize_params(n_i, n_o, 2);
	initialize_layers(10);
	size = 0;
	for (int i = 0; i < n_layers; i++) {
		size += Layers[i].W.n_elem;
	}
}

NeuralNet::NeuralNet(int n_i, int n_o, int n_l) {
	initialize_params(n_i, n_o, n_l);
	initialize_layers(10);
	size = 0;
	for (int i = 0; i < n_layers; i++) {
		size += Layers[i].W.n_elem;
	}
}

NeuralNet::NeuralNet(int n_i, int n_o, int n_l, vec ns) {
	initialize_params(n_i, n_o, n_l);
	initialize_layers(ns);
	size = 0;
	for (int i = 0; i < n_layers; i++) {
		size += Layers[i].W.n_elem;
	}
}

NeuralNet::NeuralNet(int n_i, int n_o, int n_l, vec ns, int nlin) {
	initialize_params(n_i, n_o, n_l);
	initialize_layers(ns, nlin);
	size = 0;
	for (int i = 0; i < n_layers; i++) {
		size += Layers[i].W.n_elem;
	}
}

NeuralNet::NeuralNet(int n_i, int n_o, int n_l, vec ns, int nlin, mat ws[]) {
	initialize_params(n_i, n_o, n_l);
	initialize_layers(ns, ws, nlin);
	size = 0;
	for (int i = 0; i < n_layers; i++) {
		size += Layers[i].W.n_elem;
	}

}
NeuralNet::NeuralNet(int n_i, int n_o, int n_l, vec ns, mat ws[]) {
	initialize_params(n_i, n_o, n_l);
	initialize_layers(ns, ws);
	size = 0;
	for (int i = 0; i < n_layers; i++) {
		size += Layers[i].W.n_elem;
	}
}

NeuralNet::NeuralNet(int n_i, int n_o, int n_l, int nlin) {
	initialize_params(n_i, n_o, n_l);
	initialize_layers(4, nlin);
	size = 0;
	for (int i = 0; i < n_layers; i++) {
		size += Layers[i].W.n_elem;
	}
}

NeuralNet::NeuralNet(int n_i, int n_o, int n_l, vec ns, vec nlin) {
	initialize_params(n_i, n_o, n_l);
	initialize_layers(ns, nlin);
	size = 0;
	for (int i = 0; i < n_layers; i++) {
		size += Layers[i].W.n_elem;
	}
}


void NeuralNet::initialize_params(int n_i, int n_o, int n_l) {
	input_size = n_i;
	output_size = n_o;
	n_layers = n_l;
	Layers = new Layer[n_l];
	ballSize = 1;
	size = 0;
	for (int i = 0; i < n_layers; i++) {
		size += Layers[i].W.n_elem;
	}
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

void NeuralNet::initialize_layers(int n, int nlin) {
	if (n_layers == 1) {
		Layers[0] = Layer(output_size, nlin, input_size);
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
		Layers[n_layers - 1] = Layer(output_size, nlin, n);
		last->set_next(&Layers[n_layers - 1]);
		Layers[n_layers - 1].set_previous(last);
	}
}

void NeuralNet::initialize_layers(vec ns, vec nlin) {
	if (n_layers == 1) {
		Layers[0] = Layer(output_size, nlin[0], input_size);
	}
	else {
		Layers[0] = Layer(ns[0], int(nlin[0]), input_size);
		Layer* last = &Layers[0];
		Layer* next;
		Layer* previous;
		Layer new_node;
		for (int i = 1; i < n_layers - 1; i++) {
			Layers[i] = Layer(ns[i], int(nlin[i]), ns[i-1]);
			next = &Layers[i];
			last->set_next(next);
			previous = last;
			Layers[i].set_previous(previous);
			last = &Layers[i];
		}
		Layers[n_layers - 1] = Layer(output_size, int(nlin[n_layers-1]), ns[n_layers-2]);
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
			Layers[i] = Layer(ns[i], 3, ns[i - 1]);
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

void NeuralNet::set_TRM_parameters(float lowerbound, float upperbound, float sh, float gr, double reg) {
	lb = lowerbound;
	ub = upperbound;
	shrink = sh;
	grow = gr;
	regular = reg;
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
		last_dz = (current->back_prop(last_dz,regular));
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

void NeuralNet::train_GD(mat input, mat expected_output, int n_steps, float lr, bool print, string filename, bool times, int max_time) {
	bool log;
	ofstream file, file_time;
	if (filename == "") { 
		log = false; 
	}
	else {
		file.open(filename, ios::out);
		log = true;
		if (times) {
			file_time.open("times_"+filename, ios::out);
		}
	}
	mat out;
	mat errors;
	float error;
	float last_error = INFINITY;
	if (times) {
		clock_t initial_time = clock();
		int steps = 0;
		while(steps <= n_steps && clock() - initial_time < max_time){
			out = forward_prop(input);
			errors = out - expected_output;

			back_prop(errors);
			step(lr);
			if (print) {
				error = dot(errors, errors);
				std::cout << "error: " << error << std::endl;
				if (log) {
					file << error << endl;
				}
				if (times) {
					file_time << (clock() - initial_time) << endl;
				}
			}
			steps++;

		}
	}
	else {
		for (int i = 0; i < n_steps; i++) {
			out = forward_prop(input);
			errors = out - expected_output;

			back_prop(errors);
			step(lr);
			if (print) {
				error = dot(errors, errors);
				std::cout << "error: " << error << std::endl;
				if (log) {
					file << error << endl;
				}
			}

		}
	}
	if (filename != "") {
		file.close();
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
	return float(accu(dot(errors,errors)));
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

void NeuralNet::train_TRM_cd_pu(mat input, mat expected_output, int n_steps, float bs, bool print, string filename, bool times, int max_time) {
	// maybe we should actually compute the Hessian matrix
	int i = 1;
	int random_idx;
	vec V(size);
	vec g(size);
	V.fill(0.0f);
	ballSize = bs;

	for (int l = 0; l < n_layers; l++) {
		Layers[l].BallSizes.fill(ballSize);
	}
	bool log;
	ofstream file, file_times;
	if (filename == "") {
		log = false;
	}
	else {
		file.open(filename, ios::out);
		log = true;
		if (times) {
			file_times.open("times_" + filename, ios::out);
		}
	}
	bool chose_p0 = false;
	double max_ballSize = 1000000000000;
	double d1;
	double d2;
	vec h(size);
	vec hessian_col;
	mat errors;
	int layer, l_idx;
	double approximate_delta, actual_delta, row;
	bool no_step;
	double starting_ballSize;
	double p0, p1, p_star, lambda_star, y1, y2, error1, error0, next_error;
	random_idx = 0;

	clock_t initial_time = clock();
	int flat_count;
	while (i < n_steps) { // get better stopping condition
		chose_p0 = false;
		if (i%size == 1) {
			errors = forward_prop(input) - expected_output;
			error = dot(errors, errors);
			back_prop(errors);
			int count = 0;
			int l = 0;
			while (l < n_layers) {
				for (int j = 0; j < Layers[l].W.n_elem; j++) {
					g(count) = Layers[l].GradW(j);
					//mmm i should make a mapping 
					if (count == random_idx) {
						layer = l;
						l_idx = j;
					}
					count++;
				}
				l++;
			}
			vec ec(size);
			ec.fill(0.0);
			for (int i = 0; i < size; i++) {
				ec.at(i) = 1.0;
				h(i) = dot(ec,(Hv(ec))) + regular;
			}
		}
		int rand_idx = i % size;

		d1 = g(rand_idx);
		d2 = h(rand_idx);
		bool computed_p0 = false;
		no_step = true;
		//	while (no_step) {
		if (d2 > 0) {
			// only calculate p0 if it's a min not a max
			p0 = -d1 / d2;
			if (p0 < Layers[layer].BallSizes(l_idx)) {
				computed_p0 = true;
				no_step = false;
			}
		}


		//p1 = ballSize*(d1 / abs(d1))*(d2 / abs(d2));
		p1 = -Layers[layer].BallSizes(l_idx)*(d1 / abs(d1));


		//cout << "p1:" << p1 << "ballsize:" << ballSize << endl;

		if (computed_p0) {
			Layers[layer].W(l_idx) += p1;
			errors = forward_prop(input) - expected_output;
			error1 = dot(errors, errors);
			Layers[layer].W(l_idx) += (p0 - p1);
			errors = forward_prop(input) - expected_output;
			error0 = dot(errors, errors);

			if (error0 <= error1) {
				p_star = p0;
				chose_p0 = true;
				next_error = error0;
			}
			else {
				Layers[layer].W(l_idx) += (p1 - p0);
				p_star = p1;
				next_error = error1;
			}
		}
		else {
			Layers[layer].W(l_idx) += p1;
			errors = forward_prop(input) - expected_output;
			next_error = dot(errors, errors);
			p_star = p1;

		}

		approximate_delta = p_star*d1 + (1.0 / 2.0)*p_star*p_star*d2;
		actual_delta = next_error - error;

		row = actual_delta / approximate_delta;
		if (row > ub) {
			Layers[layer].BallSizes(l_idx) *= grow;
			if (Layers[layer].BallSizes(l_idx)*grow > max_ballSize) {
				Layers[layer].BallSizes(l_idx) = max_ballSize;
			}
			no_step = false;
			//cout << "grew ";
		}
		else if (row < lb) {
			Layers[layer].W(l_idx) -= p_star;
			Layers[layer].BallSizes(l_idx) *= shrink;
			next_error = error;
		}
		else {
			no_step = false;
		}
		//		}

		if (isnan(next_error)) {
			cout << "we've got a problem" << endl;
		}

		if (next_error > error) {
			cout << "error increased" << endl;
		}
		i++;

		if (print) {
			if (chose_p0) {
				std::cout << "error: " << next_error << " p1:" << p1 << " chose p0: " << chose_p0 << " p0 " << p0 << std::endl;
			}
			else {
				std::cout << "error: " << next_error << " p1:" << p1 << " chose p0: " << chose_p0 << std::endl;
			}
			//print_weights();
			//print_grad();
			if (log) {
				file << next_error << endl;
			}
			//std::cout << "time: " << (clock() - initial_time);
			if (times) {
				file_times << (clock() - initial_time) << endl;
			}
		}
	}

}

void NeuralNet::train_TRM_cd(mat input, mat expected_output, int n_steps, float bs, bool print, string filename , bool times, int max_time) {
	// maybe we should actually compute the Hessian matrix
	int i = 0;
	int random_idx;
	vec V(size);
	vec g(size);
	V.fill(0.0f);
	ballSize = bs;

	for (int l = 0; l < n_layers; l++){
		Layers[l].BallSizes.fill(ballSize);
	}
	bool log;
	ofstream file, file_times;
	if (filename == "") {
		log = false;
	}
	else {
		file.open(filename, ios::out);
		log = true;
		if (times) {
			file_times.open("times_" + filename, ios::out);
		}
	}
	bool chose_p0 = false;
	double max_ballSize = 1000000000000;
	double d1;
	double d2;
	vec hessian_col;
	mat errors;
	int layer, l_idx;
	double approximate_delta, actual_delta, row;
	bool no_step;
	double starting_ballSize;
	double p0, p1, p_star, lambda_star, y1, y2, error1, error0, next_error;
	random_idx = 0;

	clock_t initial_time = clock();
	int flat_count;
	while (i < n_steps) { // get better stopping condition
		chose_p0 = false;
		errors = forward_prop(input) - expected_output;
		error = dot(errors, errors);
		back_prop(errors);
		d2 = 0;
		d1 = 0;
		l_idx = -1;
		layer = -1;
		flat_count = 0;
		while (abs(d2*d1) < 0.00000000000000001) {
			flat_count++;
			random_idx = rand() % size;// size - 1 - (random_idx + 1) % size;// rand() % size;
			int count = 0;
			int l = 0;
			while (l < n_layers) {
				for (int j = 0; j < Layers[l].W.n_elem; j++) {
					g(count) = Layers[l].GradW(j);
					//mmm i should make a mapping 
					if (count == random_idx) {
						layer = l;
						l_idx = j;
					}
					count++;
				}
				l++;
			}

			d1 = g(random_idx);
			V.at(random_idx) = 1;
			hessian_col = Hv(V);
			d2 = hessian_col.at(random_idx) + regular;
			if (flat_count > size) {
				cout << "probably all flat" << endl;
				return;
			}
		}

		bool computed_p0 = false;
		no_step = true;
		starting_ballSize = ballSize;
	//	while (no_step) {
		if (d2 > 0) {
			// only calculate p0 if it's a min not a max
			p0 = -d1 / d2;
			if (p0 < Layers[layer].BallSizes(l_idx)) {
				computed_p0 = true;
				no_step = false;
			}
		}

		
		//p1 = ballSize*(d1 / abs(d1))*(d2 / abs(d2));
		p1 = -Layers[layer].BallSizes(l_idx)*(d1 / abs(d1));
		

		//cout << "p1:" << p1 << "ballsize:" << ballSize << endl;

		if (computed_p0) {
			Layers[layer].W(l_idx) += p1;
			errors = forward_prop(input) - expected_output;
			error1 = dot(errors, errors);
			Layers[layer].W(l_idx) += (p0 - p1);
			errors = forward_prop(input) - expected_output;
			error0 = dot(errors, errors);

			if (error0 <= error1) {
				p_star = p0;
				chose_p0 = true;
				next_error = error0;
			}
			else {
				Layers[layer].W(l_idx) += (p1 - p0);
				p_star = p1;
				next_error = error1;
			}
		}
		else {
			Layers[layer].W(l_idx) += p1;
			errors = forward_prop(input) - expected_output;
			next_error = dot(errors, errors);
			p_star = p1;

		}

		approximate_delta = p_star*d1 + (1.0 / 2.0)*p_star*p_star*d2;
		actual_delta = next_error - error;

		row = actual_delta / approximate_delta;
		if (row > ub) {
			Layers[layer].BallSizes(l_idx) *= grow;
			if (Layers[layer].BallSizes(l_idx)*grow > max_ballSize) {
				Layers[layer].BallSizes(l_idx) = max_ballSize;
			}
			no_step = false;
			//cout << "grew ";
		}
		else if (row < lb) {
			Layers[layer].W(l_idx) -= p_star;
			Layers[layer].BallSizes(l_idx) *= shrink;
			next_error = error;
		}
		else {
			no_step = false;
		}
//		}

		if (isnan(next_error)) {
			cout << "we've got a problem" << endl;
		}
		
		if (next_error > error) {
			cout << "error increased" << endl;
		}
		i++;

		if (print) {
			if (chose_p0) {
				std::cout << "error: " << next_error << " p1:" << p1 << " chose p0: " << chose_p0 << " p0 " << p0 << std::endl;
			}
			else {
				std::cout << "error: " << next_error << " p1:" << p1 << " chose p0: " << chose_p0 << std::endl;
			}
			//print_weights();
			//print_grad();
			if (log) {
				file << next_error << endl;
			}
			//std::cout << "time: " << (clock() - initial_time);
			if (times) {
				file_times << (clock() - initial_time) << endl;
			}
		}
	}

}


void NeuralNet::train_TRM(mat input, mat expected_output, int n_steps, float bs, bool print, string filename, bool times, int max_time) {
	mat out;
	mat errors;
	float max_ball_size = 100000;
	double min_ball_size = 0.0000000000000001;
	ballSize = bs;
	ofstream file, file_times;
	bool log;
	float n1 = 0.10, n2 = 0.25, n3 = 0.25;
	if (filename == "") {
		log = false;
	}
	else {
		file.open(filename, ios::out);
		log = true;
		if (times) {
			file_times.open("times_" + filename, ios::out);
		}
	}
	double next_error;
	//cout << "forward prop" << endl;
	out = forward_prop(input);
	errors = out - expected_output;
	//cout << "back prop" << endl;
	back_prop(errors);
	error = dot(errors,errors);
	float row_k_f, model_s;
	vec p1(size);
	vec g(size);
	vec p_star;
	float model_s0;
	double actual_change;
	int i = 0;
	//float sigma = 62.5978;
	vec p0;
	vec hv;
	mat H(size,size);
	float p1_error, p0_error;
	clock_t initial_time = clock();
	bool chose_p0;
	double eig;
	while (i < n_steps ){
		//cout << "forward prop" << endl;
		chose_p0 = false;
		out = forward_prop(input);
		errors = out - expected_output;
		//cout << "back prop" << endl;
		back_prop(errors);
		//error = float(accu((errors) % (errors)));
		//cout << "get p1";
		eig = get_p1_TRM(p1, g);
		//cout << "done " << endl;
		hv = Hv(p1,regular);
		model_s = dot(g, p1) + 0.5*dot(p1, hv); // (fx - (fx + gTp + 0.5pHp)) (predicted change)
		p_star = p1;
		// I will reinstate the below section once I've replaced cg with a solver that works on indefinite matrices - actually don't need it p0 is only a local min if it's positive semidefinite
		add_p(p1);                                                      // +p1
		out = forward_prop(input);										// +p1	
		errors = out - expected_output;									// +p1				
		add_p(-p1);														// +p1
		p1_error = dot(errors, errors);
		errors = forward_prop(input) - expected_output;
		p0 = cg(g);
		p_star = p1;
		if (accu(abs(p0)) < ballSize) {

			hv = Hv(p0,regular);
			model_s0 = dot(g, p0) + 0.5*dot(p0, hv);
			add_p(p0);													// +p0
			errors = forward_prop(input) - expected_output;				// +p0
			add_p(-p0);													// +p0	
			p0_error = dot(errors, errors);
			errors = forward_prop(input) - expected_output;
			if (p0_error < p1_error) {
				//cout << "picked p0" << endl;
				p_star = p0;
				chose_p0 = true;
				model_s = model_s0;
				next_error = p0_error;
			}
			else {
				next_error = p1_error;
				//cout << "picked p1" << endl;
			}
		}
		else {
			next_error = p1_error;
			//cout << "picked p1" << endl;

		}

		if (isnan(next_error)) {
			cout << "error is nan! " << endl;
			i = n_steps;
		}

		actual_change = next_error - error;
		row_k_f = actual_change / model_s; // ratio of actual change and predicted change


		if (model_s >  0.000000000001) {
//				cout << "something's wrong" << endl;	

		}

		vec v_empty(size);
		v_empty.fill(0.0f);
		for (int i = 0; i < size; i++) {
			v_empty(i) = 1.0f;
			H.col(i) = Hv(v_empty, regular);
			v_empty(i) = 0.0f;
		}

		double sumH = accu(abs(H));
		double signed_sumH = accu(H);

		double golden_ratio = abs(H(0, 0))*ballSize*ballSize / (g(0)*g(0));
		bool stepped = true;
			
		if (row_k_f <= lb) {//|| model_s > 0) {
			ballSize = ballSize*shrink;
			next_error = error;
		/*	print_weights();
			print_grad();
			// we can also get H fully
		*/	vec v_empty(size);
			v_empty.fill(0.0f);
			for (int i = 0; i < size; i++) {
				v_empty(i) = 1.0f;
				H.col(i) = Hv(v_empty, regular);
				v_empty(i) = 0.0f;
			}
			stepped = false;
		//	H.print();
			//cout << "shrunk" << endl;
		}
		else if (row_k_f >= ub && ballSize*grow < max_ball_size) {
			add_p(p_star);
			ballSize = ballSize*grow;
			//cout << "grew" << endl;
		}
		else {
			add_p(p_star);
		}
		if (next_error > error) {
			add_p(-p_star);
			//add_p(ballSize*g / sqrt(dot(g, g)));
			next_error = error;
			stepped = false;
			ballSize = ballSize*shrink;
		}
		//Layers[0].zl.print();
		if (print) {
			if (chose_p0) {
				//std::cout << "error: " << next_error << " p1:" << p1.at(0) << " chose p0: " << chose_p0 << " p0 " << p0.at(0) << " ball: " << ballSize << "sumH" << sumH << " signed_sumH: " << signed_sumH << std::endl;
				cout << "error: " << next_error << " ball: " << ballSize <<  " eig: " << eig << " stepped: " << stepped <<  endl;
			}
			else {
				cout << "error: " << next_error << " ball: " << ballSize << " eig: " << eig << " stepped: " <<  stepped << endl;
				//std::cout << "error: " << next_error << " p1:" << p1.at(0) << " chose p0: " << chose_p0 << " ball: " << ballSize << " eig: " << eig << "sumH" << sumH << " signed_sumh: " << signed_sumH << std::endl;
			}
		
			if (log) {
				file << next_error << endl;
			}

			if (times) {
				file_times << (clock() - initial_time) << endl;
			}
			//std::cout << "time: " << (clock() - initial_time);
		}



		error = next_error;
		if (ballSize < min_ball_size) {
			i = n_steps;
		}
		i += 1;
	}
	if (filename != "") {
		cout << "closing " << filename << endl;
		file.close();
	}

	
}


void NeuralNet::add_p(vec p_star) {
	int l = 0;
	int n;
	int current = 0;
	while (l < n_layers) {
		n = Layers[l].W.n_elem;
		Layers[l].step_TRM(p_star.subvec(current, current + n - 1));
		current += n;
		l++;
	}

}

vec NeuralNet::cg(vec g, double lambda) {
	vec x(g.n_elem);
	x.randu();
	x = x / g.n_elem;
	vec r = -g - (Hv(x) + lambda*x);
	vec hv;
	vec p = r;
	double rsold = dot(r, r);
	double alpha, rsnew;

	//cout << rsold << endl;

	for (int i = 0; i < g.n_elem; i++) {
		hv = (Hv(p) + lambda*p);
		alpha = rsold / (dot(p, hv));
		x = x + alpha * p;
		r = r - alpha * hv;
		rsnew = dot(r, r);
		p = r + (rsnew / rsold) * p;
		rsold = rsnew;
		//	cout << rsold << endl;
	}

//	cout << "cg" << rsold;

	return x;
}

vec NeuralNet::cg(vec g) {
	vec x(g.n_elem);
	x.randu();
	x = x / g.n_elem;
	vec r = -g - Hv(x);
	vec hv;
	vec p = r;
	double rsold = dot(r, r);
	double alpha, rsnew;

	//cout << rsold << endl;

	for (int i = 0; i < g.n_elem; i++) {
		hv = Hv(p);
		alpha = rsold / (dot(p, hv));
		x = x + alpha * p;
		r = r - alpha * hv;
		rsnew = dot(r, r);
		p = r + (rsnew / rsold) * p;
		rsold = rsnew;
	//	cout << rsold << endl;
	}

//	cout << "cg" << rsold;

	return x;
}

vec NeuralNet::get_weights() {
	int count = 0;
	int l = 0;
	vec weights(size);
	while (l < n_layers) {
		for (int j = 0; j < Layers[l].W.n_elem; j++) {
			weights(count) = Layers[l].W(j);
			count++;
		}
		l++;
	}

	return weights;
}

void NeuralNet::set_weights(vec weights) {
	int l = 0;
	int count = 0;
	while (l < n_layers) {
		for (int j = 0; j < Layers[l].W.n_elem; j++) {
			Layers[l].W(j) = weights(count);
			count++;
		}
		l++;
	}
}

double NeuralNet::get_p1_TRM(vec &p_star, vec &g) {
	// it's all about getting info into little vectors and stuff.... oh goodness
	// vecs can keep expanding i think - look it up

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
	//cout << "...";
	if (isnan(eigvec.at(0))) {
		cout << "we've got a problem" << endl;
	}
	float eigvalue = power_series(eigvec, g);
	/*if (eigvalue < 0) {
		cout << "NEGATIVE" << endl;
		cout << eigvalue << endl;
	}
	else*/
	if (eigvalue > 0){
		//cout << "POSTIVE" << endl;
		//multiplier = -1.0;

		float neg_eigvalue = power_series(eigvec, g, -eigvalue) + eigvalue;
		//cout << "nb"; //for notbiggest
		if (neg_eigvalue > 0) {
			p_star = cg(g);
			//cout << "noneg";
//			cout << "WHATTT" << endl;
			return 1;
		}
		else {
			//cout << "second " << neg_eigvalue << endl;
		}
		
		eigvalue = neg_eigvalue;
//		cout << eigvalue << endl;
		/*
		p_star = cg(g, eigvalue);
		if (isnan(dot(p_star, p_star))) {
			cout << "IS BAD" << endl;
			p_star = -ballSize*g / sqrt(dot(g, g));
		}
		return;*/
	}
	else {
		//cout << "first: " << eigvalue << endl;
	}
	
	//cout << "power series complete" << endl;
	//eigvec = eigvec / sqrt(dot(eigvec, eigvec));
	vec Y1 = eigvec.subvec(0, size - 1);
	double size_y1 = sqrt(dot(Y1,Y1));
	if (size_y1 < (0.0001)) {
		
		cout << "HARD CASE DETECTED y1: " << size_y1<< endl;
	}
//	cout << "y2: " << size_y1 << endl;
	vec Y2 = eigvec.subvec(size, size * 2 - 1);
	double size_y2 = sqrt(dot(Y2,Y2));
//	cout << "y1: " << size_y2 << endl;
	
	if ((size_y1 / (size_y1 + size_y2))<(0.0004) || ((size_y2 / (size_y2 + size_y1)) < (0.0004))) {
		
		//cout << "HARD CASE DETECTED " << endl;
		p_star = -ballSize*g / sqrt(dot(g, g));
	}
	
	//cout << "y1: " << Y1.at(0) << " y2: " << Y2.at(0) << endl;
	//cout << "y1/y2 " << Y1.at(0) / Y2.at(0) << " should be |d1|/ballsize " << " d1 is : " << g.at(0) << " ballsize: " << ballSize << "expected ratio: " << abs(g.at(0)) / ballSize << endl;

	//vec zero = M_squiggle_v(eigvec, g, eigvalue);
	p_star = - ballSize*ballSize*Y1 / (dot(g,Y2));
	//p_star = -ballSize*(dot(g, Y2) / abs(dot(g, Y2)))*(Y1 / sqrt(dot(Y1, Y1)));
	
	double discrepency = sqrt(dot(p_star, p_star));
	//cout << discrepency;

	//p_star = p_star2;
	
	if (isnan(p_star(0))) {
		cout << "STOP HERE" << endl;
	}
	float a = dot(g, Y2);
	
	vec sign_of_g(size);
	for (int i = 0; i < size; i++) {
		if (g(i) > 0) {
			sign_of_g(i) = 1;
		}
		else {
			sign_of_g(i) = -1;
		}
	}

//	p_star = -sign_of_g % abs(p_star);


	if (isinf(dot(p_star, p_star)) || isnan(dot(p_star,p_star))) {
		cout << "bad" << endl;
	}

	return eigvalue;
	
}

float NeuralNet::power_series(vec &eigvec, vec g, double shift) {
	vec w(eigvec.n_elem);
	eigvec.randu();
	float eigvalue;
	w = Mv(eigvec, g, shift);
	vec r;
	float amount = INFINITY;
	int i = 0;
	float error, previous_error;
	bool prev_defined = false;

	eigvec = eigvec / sqrt(dot(eigvec, eigvec));
	//w.print();
	w = Mv(eigvec, g, shift);
	r = eigvec*dot(eigvec, w) - Mv(eigvec, g, shift);
	vec alt_r(size * 2);
	double alt_e;
	vec initial_r = r;
	previous_error = dot(r, r);
	//w.print();
	//eigvec.print();
	float diff_error = -1.0f;
	double stopping_amount = 0.01;
	double stopping_check = stopping_amount + 1;
	while (stopping_check > stopping_amount && i < 1000) {//need to test out different stopping criterioni < eigvec.n_elem * 100){// && (i < eigvec.n_elem) ){// || diff_error < 0)) {
		eigvec = w / sqrt(accu(w%w));
		w = Mv(eigvec, g, shift);
		if (isnan(dot(w, w))) {
			cout << "is nan" << endl;
		}
		i++;
		r = eigvec*dot(eigvec, w) - w;
		alt_r = (abs(eigvec) / sqrt(dot(eigvec, eigvec))) - (abs(w) / sqrt(dot(w, w)));
		alt_e = dot(alt_r, alt_r);
		error = dot(r, r);
		diff_error = error - previous_error;
		//cout << "diff e: " << diff_error << endl;
		previous_error = error;
		eigvalue = dot(eigvec, w);
		stopping_check = sqrt(dot(r, r)) / (abs(eigvalue)*sqrt(dot(eigvec, eigvec)));
		//cout << alt_e << endl;
	}

	eigvalue = dot(eigvec, w);
	vec prod = Mv(eigvec, g, shift);
	vec prod2 = eigvec*eigvalue;
	//cout << "residual: " << accu(abs(prod - prod2)) << endl;
	if (isnan(eigvalue)) {
		cout << "STOP" << endl;
	}
	return eigvalue;
}

float NeuralNet::power_series(vec &eigvec, vec g) {
	vec w(eigvec.n_elem);
	eigvec.randu();
	float eigvalue;
	w = Mv(eigvec, g);
	vec r;
	float amount = INFINITY;
	int i = 0;
	float error, previous_error;
	bool prev_defined = false;

	eigvec = eigvec / sqrt(dot(eigvec, eigvec));
	//w.print();
	w = Mv(eigvec, g);
	r = eigvec*dot(eigvec, w) - Mv(eigvec, g);
	vec alt_r(size * 2);
	double alt_e;
	vec initial_r = r;
	previous_error = dot(r, r);
	//w.print();
	//eigvec.print();
	float diff_error = -1.0f;
	double stopping_amount = 0.001;
	double stopping_check = stopping_amount + 1;
	while (stopping_check > stopping_amount && i < 100){//need to test out different stopping criterioni < eigvec.n_elem * 100){// && (i < eigvec.n_elem) ){// || diff_error < 0)) {
		eigvec = w / sqrt(accu(w%w));
		w = Mv(eigvec, g);
		if (isnan(dot(w,w))) {
			cout << "is nan" << endl;
		}
		i++;
		r = eigvec*dot(eigvec, w) - w;
		alt_r = (abs(eigvec) / sqrt(dot(eigvec, eigvec))) - (abs(w) / sqrt(dot(w, w)));
		alt_e = dot(alt_r, alt_r);
		error = dot(r, r);
		diff_error = error - previous_error;
		//cout << "diff e: " << diff_error << endl;
		previous_error = error;
		eigvalue = dot(eigvec, w);
		stopping_check = sqrt(dot(r, r)) / (abs(eigvalue)*sqrt(dot(eigvec, eigvec)));
		//cout << alt_e << endl;
	}

	eigvalue = dot(eigvec, w);
	vec prod = Mv(eigvec, g);
	vec prod2 = eigvec*eigvalue;
	//cout << "residual: " << accu(abs(prod - prod2)) << endl;
	if (isnan(eigvalue)) {
		cout << "STOP" << endl;
	}
	return eigvalue;
}

vec NeuralNet::M_squiggle_v(vec v, vec g, float lambda) {
	vec y1(v.n_elem / 2);
	vec y2(v.n_elem / 2);
	y1 = v.subvec(0, (v.n_elem / 2) - 1);
	y2 = v.subvec(v.n_elem / 2, v.n_elem - 1);
	vec hv(v.n_elem);
	hv.subvec(0, (v.n_elem / 2) - 1) = -y1 + Hv(y2) + lambda*y2;            
	hv.subvec((v.n_elem / 2), v.n_elem - 1) = Hv(y1) + lambda*y1 -g*(g.t()*y2) / (ballSize*ballSize);
	return hv;
}

vec NeuralNet::Mv(vec v, vec g, double shift) {
	vec y1(v.n_elem / 2);
	vec y2(v.n_elem / 2);
	y1 = v.subvec(0, (v.n_elem / 2) - 1);
	y2 = v.subvec(v.n_elem / 2, v.n_elem - 1);
	vec hv(v.n_elem);
	hv.subvec(0, (v.n_elem / 2) - 1) = Hv(y1, regular) - g*(dot(g, y2)) / (ballSize*ballSize);
	hv.subvec((v.n_elem / 2), v.n_elem - 1) = -y1 + Hv(y2, regular);
	
	return hv + shift*v;
}

vec NeuralNet::Mv(vec v, vec g) {
	vec y1(v.n_elem / 2);
	vec y2(v.n_elem / 2);
	y1 = v.subvec(0,(v.n_elem / 2)-1);
	y2 = v.subvec(v.n_elem / 2, v.n_elem - 1);
	vec hv(v.n_elem);
	hv.subvec(0, (v.n_elem / 2) - 1) = Hv(y1,regular) - g*(dot(g,y2))/(ballSize*ballSize);
	hv.subvec((v.n_elem / 2), v.n_elem - 1) = -y1 + Hv(y2,regular);
	return hv;
}

vec NeuralNet::Hv(vec v, double reg) {
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
		//cout << "forward proping HV" << endl;
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
		//cout << "backward propping HV" << endl;
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

	return hv + v*reg;
}

void NeuralNet::print_grad() {
	for (int i = 0; i < n_layers; i++) {
		Layers[i].print_grad();
	}
}

