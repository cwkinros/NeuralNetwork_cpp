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
	void step_TRM(vec p_star);
	vec g	 (vec input);
	vec g1	 (vec input, int i, vec dz);
	mat g2(mat inputs);
	mat g(mat input);
	mat output(mat input);
	mat back_prop(mat input, double reg = 1.0f);
	
	mat forwardHv(mat R_input, mat V);
	mat backHv(mat R_dz, mat V);
	void set_next(Layer* l);
	void set_previous(Layer* l);
	double singleHv(int i, int j);
	Layer* get_next();
	Layer* get_previous();
	mat R_dw; // this is Hv for this layer
	mat W;
	mat GradW;
	mat BallSizes;
	mat zl;
private:
	int n;
	int non_lin;
	mat Inputs;
	mat hs;
	mat dz;
	mat g2_hs;
	mat g1_hs;
	mat R_hs;
	mat R_input;
	mat R_dz;
	mat R_dh;
	mat dzg1hs;
	Layer* Next;
	Layer* Previous;
};