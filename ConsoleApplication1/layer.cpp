#include "layer.h"
#include "math_utils.h"
#include <iostream>

using namespace std;


Layer::Layer(int num, int type, int input_size) {
	n = num;
	non_lin = type;
	Matrix Wmat(input_size, num, 1.0f);
	W = Wmat;
	Matrix Gmat(input_size, num, 0.0f);
	GradW = Gmat;
	Vector v(input_size);
	Next = NULL;
	Previous = NULL;
}

void Layer::step(float step_size) {
	W.add_ip(GradW.multiply(step_size));
	//W.print();
	//GradW.print();
}

Layer::Layer(int num, int type, Matrix w, Matrix gradw) {
	n = num;
	non_lin = type;
	W = w;
	GradW = gradw;
	Vector v(w.get_c());
	Next = NULL;
	Previous = NULL;
}

Layer::~Layer() {
	std::cout << "DELETING LAYER" << std::endl;
}

Matrix Layer::back_prop(Matrix error) {
	GradW.set_all(0.0f);
	Vector dh;
	Matrix grad;
	Vector *next_errors = new Vector[error.get_c()];
	for (int i = 0; i < error.get_c(); i++) {
		dh = error.get_col(i);
		/*cout << "error: " << endl;
		error.print();*/
		next_errors[i] = W.pre_multiply(dh);
		/*for (int c = 0; c < Inputs.get_r(); c++) {
			cout << "inputs at " << c << " is: " << Inputs.get_col(i).get(c) << endl;
		}
		for (int d = 0; d < dh.get_length(); d++) {
			cout << "dh at " << d << " is: " << dh.get(d) << endl;
		}*/
		grad = Matrix(Inputs.get_col(i),dh);
		/*cout << "little grad:" << endl;
		grad.print();*/
		GradW.add_ip(grad);
	}

	GradW.multiply_ip(1.0f/float(error.get_c()));
	Matrix next(error.get_c(), W.get_c(), next_errors);
	return next;
}

Vector Layer::back_prop(Vector error) {
	Vector dh = error;
	Vector next_error = W.pre_multiply(dh);
	GradW = Matrix(dh, Input);
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
	return h;
}

void Layer::set_next(Layer* l) {
	Next = l;
	int a = 1;
}

void Layer::set_previous(Layer* l) {
	Previous = l;
	int a = 1;
}

Layer* Layer::get_next() {
	return Next;
}

Layer* Layer::get_previous() {
	return Previous;
}

int Layer::get_input_n() {
	return W.get_c();
}

int Layer::get_output_n() {
	return W.get_r();
}








