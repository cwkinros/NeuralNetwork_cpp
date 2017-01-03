#pragma once
#include "math_utils.h"
class Layer {
public:
	Layer() {
		n = 0;
	};
	Layer(int n, int non_lin_type, int input_size);
	Layer(int n, int non_lin_type, Matrix W, Matrix GradWi);
	~Layer();

	int get_input_n();
	int get_output_n();
	void step(float step_size);
	Vector g(Vector input);
	Vector g1(Vector input);
	Vector output(Vector input);
	Matrix output(Matrix input);
	Vector back_prop(Vector input);
	Matrix back_prop(Matrix input);
	void set_next(Layer* l);
	void set_previous(Layer* l);
	Layer* get_next();
	Layer* get_previous();

private:
	int n;
	int non_lin;
	Matrix W;
	Matrix GradW;
	Vector Input;
	Matrix Inputs;
	Layer* Next;
	Layer* Previous;


};