#include "math_utils.h"
#include "stdafx.h"
#include <iostream>


Matrix::Matrix() = default;
// helper function

Vector* initialize_matrix(int n_rows, int n_cols) {
	Vector* matrix = new Vector[n_cols];
	for (int i = 0; i < n_cols; i++) {
		matrix[n_cols] = Vector(n_rows);
	}
	return matrix;
}

Matrix::Matrix(int c, int r) {
	rows = r;
	cols = c;

	values = new Vector[c];
}

Matrix::Matrix(int c, int r, float val) {
	rows = r;
	cols = c;
	values = new Vector[c];
	Vector v(r, val);
	values[0] = v;
	for (int i = 1; i < c; i++) {
		values[i] = v.copy();
	}
}

Matrix::Matrix(Vector v, Vector u) {
	cols = v.get_length();
	rows = u.get_length();
	values = new Vector[cols];
	for (int i = 0; i < cols; i++) {
		values[i] = u.multiply(v.get(i));
	}

}


Matrix::Matrix(int c, int r, Vector* vals) {
	rows = r;
	cols = c;
	values = vals;
}

Matrix::Matrix(int c, int r, float** vals) {
	rows = r;
	cols = c;
	values = new Vector[c];
	for (int i = 0; i < c; i++) {
		Vector v(r, vals[i]);
		std::cout << "initialized vector: " << std::endl;
		v.print();
		values[i] = v;
	}
}


void Matrix::set(int c, int r, float value) {
	if (c >= cols || r >= rows) { std::cerr << "Indices out of bounds" << std::endl; }
	values[c].set(r, value);
}

int Matrix::get_c() {
	return cols;
}

int Matrix::get_r() {
	return rows;
}

void Matrix::set_col(int c, Vector vals) {
	//should be a delete here
	if (vals.get_length() != rows) { std::cerr << "length of vector must be equal to #rows of matrix to set_col" << std::endl; }
	values[c] = vals;
}

float Matrix::get(int c, int r) {
	if (c >= cols || r >= rows) { std::cerr << "indices out of bounds" << std::endl; }
	return values[c].get(r);
}

Vector Matrix::get_col(int c) {
	if (c >= cols) { std::cerr << "column index is out of bounds" << std::endl; }
	return values[c];
}

Matrix Matrix::multiply(Matrix m) {
	int c = m.get_c();
	int r = m.get_r();
	float** result_ptr = new float*[c];
	
	// new matrix will be r_new = rows, c_new = c
	// note r == cols
	if (r != cols){ std::cerr << "Dimension mismatch: m has " << r << " rows, and this has " << cols << " columns." << std::endl;}

	for (int i = 0; i < c; i++) {
		result_ptr[i] = new float[rows];
		Vector col = m.get_col(i);
		for (int k = 0; k < cols; k++) {			
			for (int j = 0; j < rows; j++) {
				if (k == 0) { result_ptr[i][j] = 0; }
				result_ptr[i][j] = result_ptr[i][j] + col.get(k)*values[k].get(j);
			}
		}
	}
	Matrix product(c, rows, result_ptr);
	return product;
}

void Matrix::print() {
	for (int j = 0; j < rows; j++) {
		for (int i = 0; i < cols; i++) {
			std::cout << values[i].get(j) << " ";
		}
		std::cout << std::endl;
	}
}

Matrix Matrix::multiply(float mult) {
	Vector* vals = new Vector[cols];
	float temp;
	for (int i = 0; i < cols; i++) {
		vals[i] = Vector(rows, 0.0f);
		for (int j = 0; j < rows; j++) {
			temp = values[i].get(j)*mult;
			vals[i].set(j, temp); 
		}
	}
	Matrix m(cols, rows, vals);
	return m;

}

Matrix Matrix::multiply(int mult) {
	float multiplier = float(mult);
	return (this->multiply(multiplier));
}

void Matrix::multiply_ip(float mult) {
	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < rows; j++) {
			values[i].multiply(mult);
		}
	}
}

void Matrix::multiply_ip(int mult) {
	float multiplier = float(mult);
	this->multiply_ip(multiplier);
}

Vector Matrix::pre_multiply(Vector v)
{
	if (v.get_length() != rows) { std::cerr << "Dimension mismatch" << std::endl; }
	float* _result = new float[cols];
	for (int i = 0; i < cols; i++) {
		_result[i] = v.dot(values[i]);
	}
	Vector result(cols, _result);
	return result;
}

void Matrix::set_all(float val) {
	for (int i = 0; i < cols; i++) {
		values[i].set_all(val);
	}
}

Matrix Matrix::sub(Matrix m) {
	Vector* _result = new Vector[cols];
	for (int i = 0; i < cols; i++) {
		_result[i] = values[i].sub(m.get_col(i));
	}
	Matrix result(cols, rows, _result);
	return result;
}

void Matrix::add_ip(Matrix m) {
	if (m.get_c() != cols || m.get_r() != rows) {
		if (m.get_c() == rows && m.get_r() == cols) {
			for (int i = 0; i < cols; i++) {
				for (int j = 0; j < rows; j++) {
					values[i].set(j, values[i].get(j) + m.get(j, i));
				}
			}
		}	
		else {
			std::cerr << "dimension mismatch";
		}
	}
	else {
		for (int i = 0; i < cols; i++) {
			values[i].add_ip(m.get_col(i));
		}
	}
}

void Matrix::sub_ip(Matrix m) {
	for (int i = 0; i < cols; i++) {
		values[i].sub_ip(m.get_col(i));
	}
}

float Matrix::sum_squared() {
	float sum = 0.0;
	for (int i = 0; i < cols; i++) {
		sum = sum + values[i].sum_squared();
	}
	return sum;
}

Vector Matrix::multiply(Vector v) {

	if (v.get_length() != cols) { 
		std::cerr << "Dimension mismatch, cols: "<< cols << " vlength: " << v.get_length() << std::endl; 
	}
	float* zeros = new float[rows];
	for (int j = 0; j < rows; j++) {
		zeros[j] = 0.0f;
	}
	Vector result(rows, zeros);
	for (int i = 0; i < cols; i++) {
		result.add_ip(values[i].multiply(v.get(i)));
	}
	return result;
}





