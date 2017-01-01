#include "layer.h"
#include "math_utils.h"

using namespace std;


Layer::Layer(int num, int type, int input_size) {
	n = num;
	non_lin = type;
	Matrix Wmat(input_size, num, 1.0f);
	W = Wmat;
	Matrix Gmat(input_size, num, 0.0f);
	GradW = Gmat;
	Vector v(input_size);
	Layer empty;
	Empty = &empty;
	Next = &empty;
	Previous = &empty;
}

void Layer::step(float step_size) {
	W.add_ip(GradW.multiply(-step_size));
}

Layer::Layer(int num, int type, Matrix w, Matrix gradw) {
	n = num;
	non_lin = type;
	W = w;
	GradW = gradw;
	Vector v(w.get_c());
	Layer empty;
	*Empty = empty;
	*Next = empty;
	*Previous = empty;
}

Matrix Layer::back_prop(Matrix error) {
	GradW.set_all(0.0f);
	Vector dh;
	Matrix grad;
	Vector *next_errors = new Vector[error.get_c()];
	for (int i = 0; i < error.get_c(); i++) {
		dh = this->g1(error.get_col(i));
		next_errors[i] = W.pre_multiply(dh);
		Matrix grad(dh, Inputs.get_col(i));
		GradW.add_ip(grad);
	}

	GradW.multiply_ip(1.0f/float(error.get_c()));
	Matrix next(W.get_c(), error.get_c(), next_errors);
	return next;
}

Vector Layer::back_prop(Vector error) {
	Vector dh = this->g1(error);
	Vector next_error = W.pre_multiply(dh);
	Matrix grad(dh, Input);
	GradW = grad;
	return next_error;
}

Vector Layer::g1(Vector input) {
	switch (non_lin) {
	case (1) :
		input.multiply_ip(2);
		break;
	case (2) : 
		for (int i = 0; i < input.get_length(); i++) {
			input.set(i, 2);
		}
		break;
	default :
		for (int i = 0; i < input.get_length(); i++) {
			input.set(i, 1);
		}
		break;
	}
	return input;
}

Vector Layer::g(Vector input) {
	switch (non_lin) {
	case (1) :
		input.squared_ip();
		return input;
	case (2) : 
		input.multiply_ip(2);
		return input;
	default :
		return input;
	}
}

Matrix Layer::output(Matrix input) {
	Inputs = input;
	Vector* _result = new Vector[input.get_c()];
	for (int i = 0; i < input.get_c(); i++) {
		_result[i] = this->output(input.get_col(i));
	}
	Matrix result(input.get_c(), _result[0].get_length(), _result);
	return result;
}

Vector Layer::output(Vector input) {
	Input = input;
	Vector h = W.multiply(input);
	return this->g(h);
}

void Layer::set_next(Layer* l) {
	Next = l;
	int a = 1;
}

void Layer::set_previous(Layer* l) {
	Previous = l;
}

Layer Layer::get_next() {
	return *Next;
}

Layer Layer::get_previous() {
	return *Previous;
}

int Layer::get_input_n() {
	return W.get_c();
}

int Layer::get_output_n() {
	return W.get_r();
}








