
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

mat Layer::back_prop(mat dz) {
	GradW.fill(0.0f);
	vec dh;
	mat grad;
	mat next_dz(W.n_cols, dz.n_cols);
	for (unsigned int i = 0; i < dz.n_cols; i++) {
		dh = dz.col(i)%g1(hs.col(i),i);
		next_dz.col(i) = W.t()*dh;
		GradW += GradW + dh*Inputs.col(i).t();
	}
	// update GradW
	GradW = GradW*(1.0f / float(dz.n_cols));
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
	default:
		break;
	}
	return input;
}

vec Layer::g1(vec input, int i) {
	vec ones(zl.n_rows);
	ones.fill(1.0f);
	switch (non_lin) {
	case (1) :
		input = input * 2.0f;
		break;
	case (2) :
		input.fill(2.0f);
		break;
	case (3) :
		input = zl.col(i)%(-1*zl.col(i) + ones);
		break;
	default :
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