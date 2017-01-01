#include "math_utils.h"
#include <iostream>

Vector::Vector() = default;

Vector::Vector(int size) {
	length = size;
	values = new float[size];
	for (int i = 0; i < length; i++) {
		values[i] = 0;
	}
}

Vector::Vector(int size, float val) {
	length = size;
	values = new float[size];
	for (int i = 0; i < size; i++) {
		values[i] = val;
	}
}

Vector::Vector(int size, float* ptr) {
	length = size;
	values = ptr;
}

void Vector::set(int i, float val) {
	values[i] = val;
}

void Vector::set_all(float val) {
	for (int i = 0; i < length; i++) {
		values[i] = val;
	}
}

Vector Vector::copy() {
	float* vals = new float[length];
	for (int i = 0; i < length; i++) {
		vals[i] = values[i];
	}

	Vector copied(length, vals);
	return copied;
}


float Vector::get(int i) {
	return values[i];
}

Vector Vector::squared() {
	float* _result = new float[length];
	for (int i = 0; i < length; i++) {
		_result[i] = values[i] * values[i];
	}
	Vector result(length, _result);
	return result;
}

void Vector::squared_ip() {
	for (int i = 0; i < length; i++) {
		values[i] = values[i] * values[i];
	}
}

int Vector::get_length() {
	return length;
}

float Vector::dot(Vector v) {
	float sum = 0.0f;
	if (length == v.get_length()) {
		for (int i = 0; i < length; i++) {
			sum = sum + values[i] * v.get(i);
		}
		return sum;
	}
	// check how to call and error in c++
	return 0;
}

Vector Vector::multiply(float mult) {	
	float* new_values = new float[length];
	for (int i = 0; i < length; i++) {
		new_values[i] = mult*values[i];
	}
	Vector v(length, new_values);
	return v;
}

Vector Vector::multiply(int mult) {
	float multiplier = float(mult);
	return this->multiply(multiplier);
}

void Vector::multiply_ip(float mult) {
	for (int i = 0; i < length; i++) {
		values[i] = mult*values[i];
	}
}

void Vector::multiply_ip(int mult) {
	float multiplier = float(mult);
	return this->multiply_ip(multiplier);
}

Vector Vector::add(Vector v) {
	float* _result = new float[length];
	for (int i = 0; i < length; i++) {
		_result[i] = values[i] + v.get(i);
	}
	Vector result(length, _result);
	return result;
}

Vector Vector::sub(Vector v) {
	float* _result = new float[length];
	for (int i = 0; i < length; i++) {
		_result[i] = values[i] - v.get(i);
	}
	Vector result(length, _result);
	return result;
}

float Vector::sum_squared() {
	float sum = 0.0f;
	for (int i = 0; i < length; i++) {
		sum = sum + values[i] * values[i];
	}
	return sum;
}

void Vector::add_ip(Vector v) {
	for (int i = 0; i < length; i++) {
		values[i] = values[i] + v.get(i);
	}
}

void Vector::sub_ip(Vector v) {
	for (int i = 0; i < length; i++) {
		values[i] = values[i] - v.get(i);
	}
}

void Vector::print() {
	for (int i = 0; i < length; i++) {
		std::cout << values[i];
	}
	std::cout << std::endl;
}