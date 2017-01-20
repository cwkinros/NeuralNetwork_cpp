#pragma once
#include "math_utils.h"
#include <armadillo>

using namespace arma;

class Layer_fs {
public:
	Layer_fs() {
		n = 0;
	};
	Layer_fs(int n, int non_lin_type, int input_size);
	Layer_fs(int n, int non_lin_type, Matrix W, Matrix GradWi);
	~Layer_fs();

	int get_input_n();
	int get_output_n();
	void step(float step_size);
	Vector g(Vector input);
	Vector g1(Vector input);
	Vector output(Vector input);
	Matrix output(Matrix input);
	Vector back_prop(Vector input);
	Matrix back_prop(Matrix input);
	void set_next(Layer_fs* l);
	void set_previous(Layer_fs* l);
	Layer_fs* get_next();
	Layer_fs* get_previous();

private:
	int n;
	int non_lin;
	Matrix W;
	Matrix GradW;
	Vector Input;
	Matrix Inputs;
	Layer_fs* Next;
	Layer_fs* Previous;
};


class Layer {
public:
	Layer() {
		n = 0;
	};
	Layer(int n, int non_lin_type, int input_size);
	Layer(int n, int non_lin_type, int input_size, mat W);
	void print_weights();
	void print_grad();

	int get_input_n();
	int get_output_n();
	void step(float step_size);
	vec g	 (vec input);
	vec g1	 (vec input, int i, vec dz);
	mat g(mat input);
	mat output(mat input);
	mat back_prop(mat input);
	void set_next(Layer* l);
	void set_previous(Layer* l);
	Layer* get_next();
	Layer* get_previous();
private:
	int n;
	int non_lin;
	mat W;
	mat GradW;
	mat Inputs;
	mat hs;
	mat zl;
	Layer* Next;
	Layer* Previous;
};