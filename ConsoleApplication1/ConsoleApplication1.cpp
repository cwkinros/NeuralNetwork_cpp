// ConsoleApplication1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "math_utils.h"
#include "NN.h"
#include <iostream>
using namespace std;

void testMatrices() {
	const int cols1 = 3;
	const int rows1 = 2;

	float** matrix1 = new float*[cols1];
	int j;
	for (int i = 0; i < cols1; i++) {
		matrix1[i] = new float[rows1];
		for (j = 0; j < rows1; j++) {
			matrix1[i][j] = float(i + j);
		}
	}

	Matrix m1(cols1, rows1, matrix1);
	cout << "m1: " << endl;
	m1.print();

	int rows2 = cols1;
	int cols2 = 4;

	float** matrix2 = new float*[cols2];

	for (int i = 0; i < cols2; i++) {
		matrix2[i] = new float[rows2];
		for (j = 0; j < rows2; j++) {
			matrix2[i][j] = float(i + j);
		}
	}


	Matrix m2(cols2, rows2, matrix2);
	cout << "m2: " << endl;
	m2.print();

	Matrix m3 = m1.multiply(m2);

	cout << "m3: " << endl;
	m3.print();
}

int main()
{

	/* we want to make a two layer neural network with 2 inputs, 10 nodes in the hidden layer and 1 output
	* 

	float v1[2] = { 1.0f,2.0f };
	float o1[1] = { 3.0f };
	float v2[2] = { 2.0f,3.0f };
	float o2[1] = { 5.0f };
	Vector input1(2, v1);
	Vector input2(2, v2);
	Vector output1(1, o1);
	Vector output2(1, o2);
	Vector input_data[2] = { input1, input2 };
	Vector output_data[2] = { output1, output2 };
	Matrix inputs(2, 2, input_data);
	Matrix outputs(2, 1, output_data);

	NeuralNetwork nn(2, 1);
	
	nn.train_GD(inputs, outputs);
	*/



	std::cout << "Hello World!" << std::endl;
	system("pause");
	return 0;
}

