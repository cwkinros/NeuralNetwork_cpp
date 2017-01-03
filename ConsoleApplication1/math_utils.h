#pragma once

class Vector {
public:

	//constructors
	Vector();
	Vector(int size);
	Vector(int size, float val);
	Vector(int size, float* vals);

	//mutators
	void set(int i, float val);
	void set_all(float value);

	//accessors?
	float get(int i);
	int get_length();

	// other functions
	float dot(Vector v);
	Vector multiply(float mult);
	Vector multiply(int mult);
	void multiply_ip(float mult);
	void multiply_ip(int mult);
	Vector add(Vector v);
	Vector sub(Vector v);
	void add_ip(Vector v);
	void sub_ip(Vector v);

	void squared_ip();
	float sum_squared();
	Vector squared();


	Vector copy();
	void print();

private:
	int length;
	float* values;
};
class Matrix {
public:

	// constructors
	Matrix();
	Matrix(int c, int r);
	Matrix(int c, int r, float val);
	Matrix(Vector v, Vector u);
	Matrix(int c, int r, Vector* vals);
	Matrix(int c, int r, float** vals);

	//mutators?
	void set(int c, int r, float value);
	void set_col(int c, Vector v);
	void set_all(float value);
	float sum_squared();

	//for accessing
	float get(int c, int r);
	int get_c();
	int get_r();
	Vector get_col(int c);

	Matrix multiply(Matrix m);
	Matrix multiply(float mult);
	Matrix multiply(int mult);
	Vector multiply(Vector v);
	Vector pre_multiply(Vector v);

	Matrix sub(Matrix m);
	void add_ip(Matrix m);
	void sub_ip(Matrix m);

	// ip: in place
	void multiply_ip(float mult);
	void multiply_ip(int mult);
	void print();
	~Matrix();
private:
	int cols;
	int rows;	
	Vector* values;
};

