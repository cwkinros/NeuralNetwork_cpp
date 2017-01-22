
#include "layer.h"
#include <armadillo>
#include <iostream>

using namespace std;
using namespace arma;

Layer::Layer(int num, int type, int input_size) {
	n = num;
	non_lin = type;
	//W = mat(num,input_size);
	W = randu(num, input_size);
	W = W - 0.5f;
	W = W * 2.0f / float(sqrt(input_size));
	GradW = mat(num, input_size);
	Next = NULL;
	Previous = NULL;
}


Layer::Layer(int num, int type, int input_size, mat w) {
	n = num;
	non_lin = type;
	W = w;
	GradW = mat(num, input_size);
	Next = NULL;
	Previous = NULL;
}


void Layer::step(float step_size) {
	W = W - (GradW*step_size);
}

mat Layer::back_prop(mat this_dz) {
	g1_hs = mat(n, this_dz.n_cols);
	g2_hs = mat(n, this_dz.n_cols);
	GradW.fill(0.0f);
	vec dh;
	mat grad;
	mat next_dz(W.n_cols, this_dz.n_cols);
	for (unsigned int i = 0; i < this_dz.n_cols; i++) {
		g1_hs.col(i) = g1(hs.col(i), i, this_dz.col(i));	
		dh = this_dz.col(i)%g1_hs.col(i);
		next_dz.col(i) = W.t()*dh;
		GradW += dh*Inputs.col(i).t();
	}
	g2_hs = g2(hs);
	// update GradW
	GradW = GradW*(1.0f / float(this_dz.n_cols));
	dz = this_dz;
	//GradW.print("gradW: ");
	return next_dz;
}


vec Layer::g(vec input) {
	switch (non_lin) {
	case (1) :
		input = input * input;
		break;
	case (2) :
		input = input * 2.0f;
		break;
	case (3) :
		input = 1.0f / (1.0f + exp(-input));
		break;
	case (4) :
		input = exp(input);
		input = input / sum(input);
	default:
		break;
	}
	return input;
}

vec Layer::g1(vec input, int i, vec dz) {
	vec ones(zl.n_rows);
	ones.fill(1.0f);
	vec helper;
	switch (non_lin) {
	case (1) :
		input = input * 2.0f;
		break;
	case (2) :
		input.fill(2.0f);
		break;
	case (3) :
		input = g(input) % (ones - g(input));
		//input = zl.col(i)%(ones-zl.col(i));
		break;
	case (4) :
		input = zl.col(i)%(dz - sum(dz%zl.col(i)));
	default :
		input.fill(1.0f);
		break;
	}
	return input;
}

mat Layer::g2(mat input) {
	mat ones(zl.n_rows,zl.n_cols);
	ones.fill(1.0f);
	switch (non_lin) {
	case (1) :
		input.fill(2.0f);
		break;
	case (2) :
		input.fill(0.0f);
		break;
	case (3) :
		input = g1_hs % (ones - 2.0f*zl);
		break;
	case (4) :
		// need to fix more than just this
		input = ones;
	default:
		input.fill(0.0f);
		break;
	}
	return input;
}

mat Layer::g(mat input) {
	switch (non_lin) {
	case (1) :
		input = input * input;
		break;
	case (2) :
		input = input * 2.0f;
		break;
	case (3) :
		input = 1.0f / (1.0f + exp(-input));
		break;
	default:
		break;
	}
	return input;
}

mat Layer::output(mat input) {
	Inputs = input;
	hs = W*input;
	zl = g(hs);
	return zl;
}

mat Layer::forwardHv(mat _R_input, mat V) {

	mat _R_h(n,_R_input.n_cols);
	_R_h.fill(0.0f);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < _R_input.n_rows; j++) {
			_R_h.row(i) = _R_h.row(i) + W(i, j)*_R_input.row(i) + V(i, j)*Inputs.row(j);
		}
	}
	R_input = _R_input;
	mat R_z = g1_hs%_R_h;
	R_hs = _R_h;
	return R_z;
}

mat Layer::backHv(mat R_dz, mat V) {
	mat R_dh(n, R_dz.n_cols);
	R_dh = R_dz%g1_hs + dz%g2_hs%R_hs;
	R_dw = R_dh*Inputs.t() + (dz%g1_hs)*R_input.t();
	mat R_dz_1(Inputs.n_rows, n);
	R_dz_1 = W.t()*R_dh + V.t()*(dz%g1_hs);
	return R_dz_1;
}

void Layer::set_next(Layer* l) {
	Next = l;
}

void Layer::set_previous(Layer* p) {
	Previous = p;
}

Layer* Layer::get_next() {
	return Next;
}

Layer* Layer::get_previous() {
	return Previous;
}

int Layer::get_input_n() {
	return W.n_cols;
}

int Layer::get_output_n() {
	return W.n_cols;
}

void Layer::print_weights() {
	W.print();
}

void Layer::print_grad() {
	GradW.print();
}

