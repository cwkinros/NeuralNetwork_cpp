// ConsoleApplication1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
//#include "math_utils.h"
//#include "NN.h"
#include <iostream>
#include <vector>
#include <armadillo>
#include "NeuralNet.h"
#include <fstream>
#include <time.h>

using namespace std;
using namespace arma;



// code from https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void ReadMNIST(string filename_input, string filename_label, int NumberOfImages, int DataOfAnImage, mat &arr, vec &labels)
// code from https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/
{
	arr.resize(DataOfAnImage, NumberOfImages);
	labels.resize(NumberOfImages);
	ifstream file(filename_input, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i < number_of_images; ++i)
		{
			for (int r = 0; r<n_rows; ++r)
			{
				for (int c = 0; c<n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					arr((n_rows*r) + c, i) = (double)temp;
				}
			}
		}
	}
	file.close();

	ifstream file_labels(filename_label, ios::binary);
	if (file_labels.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		file_labels.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file_labels.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		int val;
		for (int i = 0; i < number_of_images; ++i) {
			unsigned char temp = 0;
			file_labels.read((char*)&temp, sizeof(temp));
			labels(i) = (int)temp;
		}
	}
	file_labels.close();
}


void labelvec2mat(mat &labels, vec labelvec) {
	labels.resize(10, labelvec.n_elem);
	for (int i = 0; i < labelvec.n_elem; i++) {
		vec zeros(10);
		zeros.fill(0);
		zeros(labelvec(i)) = 1;
		labels.col(i) = zeros;
	}
}

void save_file(string filename_input, string filename_output, mat arr, vec labels ,int n_cols) {
	ofstream output_file, input_file;
	output_file.open(filename_output, ios::out);
	input_file.open(filename_input, ios::out);
	if (output_file.is_open() && input_file.is_open()) {
		cout << "files are open and we are now printing" << endl;
		input_file << n_cols << " " << 784 << endl;
		output_file << n_cols << endl;
		for (int i = 0; i < n_cols; i++) {
			for (int j = 0; j < 784; j++) {
				input_file << arr(j, i) << " ";
			}
			output_file << labels(i) << " ";
			input_file << endl;
		}
	}
	else {
		cout << "files were not open to print to..." << endl;
	}
	output_file.close();
	input_file.close();
}

void from_files(string filename_input, string filename_output, mat &arr, vec &labelvec) {

	ifstream label_file, input_file;
	label_file.open(filename_output, ios::in);
	input_file.open(filename_input, ios::in);
	
	int rows_input, rows_label, cols_input, cols_label;
	input_file >> rows_input;
	input_file >> cols_input;
	label_file >> rows_label;

	int rows;
	if (rows_input < rows_label) {
		rows = rows_input;
	}
	else {
		rows = rows_label;
	}
	arr.resize(cols_input, rows);
	labelvec.resize(rows);
	int input;
	int label;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols_input; j++) {
			input_file >> input;
			while (input > 255 || input < 0) {
				input_file >> input;
			}
			arr(j, i) = input;
		}
		label_file >> labelvec(i);
	}

}

void preset_start(string filename, int n_i, int n_o, int n_l, vec ns) {
	ofstream file(filename, ios::out);
	file << n_l << endl;
	if (file.is_open()) {
		file << ns[0] << " " << n_i << endl;
		mat w1 = randu(ns[0], n_i);
		w1 = w1 - 0.5f;
		w1 = w1 * 2.0f / float(n_i);
		for (int i = 0; i < ns[0]; i++) {
			for (int j = 0; j < n_i; j++) {
				file << w1(i, j) << " ";
			}
			file << endl;
		}
		for (int l = 1; l < n_l - 1; l++) {
			file << ns[l] << " " << ns[l - 1] << endl;
			mat w = randu(ns[l], ns[l - 1]);
			w = w - 0.5f;
			w = w * 2.0f / float(ns[l - 1]);
			for (int i = 0; i < ns[l]; i++) {
				for (int j = 0; j < ns[l - 1]; j++) {
					file << w(i, j) << " ";
				}
				file << endl;
			}
		}
		file << n_o << " " << ns[n_l - 2] << endl;
		mat w = randu(n_o, ns[n_l - 2]);
		w = w - 0.5f;
		w = w * 2.0f / float(ns[n_l - 2]);
		for (int i = 0; i < n_o; i++) {
			for (int j = 0; j < ns[n_l-2]; j++) {
				file << w(i, j) << " ";
			}
			file << endl;
		}
	}
	file.close();
}

mat* get_start(string filename) {
	ifstream file(filename, ios::in);
	int n_l;
	int n_i, n_o;
	if (file.is_open()) {
		file >> n_l;
		const int n = n_l;
		mat* ws = new mat[n];
		for (int l = 0; l < n_l; l++) {
			file >> n_o;
			file >> n_i;
			ws[l] = mat(n_o, n_i);
			for (int i = 0; i < n_o; i++) {
				for (int j = 0; j < n_i; j++) {
					file >> ws[l](i, j);
				}
			}
		}
		return ws;
	}
	return NULL;
	
}

float test1_lecun(string start_filename, string finished_filename) {

	// initializing network itself
	cout << "initializing network values...." << endl;
	int n_l = 3;
	int n_i = 784;
	int n_o = 10;
	vec ns(2);
	ns << 500 << 150 << endr;
	cout << "saving weights start ... " << endl;
	preset_start(start_filename, n_i, n_o, n_l, ns);
	cout << "retrieving weights start ... " << endl;
	mat* ws = get_start(start_filename);
	cout << "initialize neural net... " << endl;
	NeuralNet nn(784, 10, 3, ns, ws);
	string filename_input = "train-images.idx3-ubyte";
	string filename_label = "train-labels.idx1-ubyte";
	// gathering training data
	mat arr, labels;
	vec labelvec;
	cout << "input training data and training labels (time consuming)" << endl;
	ReadMNIST(filename_input, filename_label, 60000, 784, arr, labelvec);
	labelvec2mat(labels, labelvec);

	// training
	cout << "begin training using GD" << endl;
	nn.train_GD(arr, labels,1000,0.1f,false,"");
	

	filename_input = "t10k-images.idx3-ubyte";
	filename_label = "t10k-labels.idx1-ubyte";
	vec labs;
	// gather test data
	cout << "gather test data..." << endl;
	ReadMNIST(filename_input, filename_label, 10000, 784, arr, labs);
	cout << "run test data through nn..." << endl;
	return nn.accuracy_test(arr, labs);
}

float mnist_2layer_SGD() {
	mat arr, labels;
	vec labelvec;
	from_files("input.txt", "output.txt", arr, labelvec);
	labelvec2mat(labels, labelvec);

	arr = arr / (255.0f);
	arr = arr - 0.5f;

	vec ns(2);
	ns << 100 << 50 << endr;


	NeuralNet nn(784, 10, 2, ns, 4);
	nn.train_SGD(arr, labels, 10000, 10.0f , 10000.0f, true, "mnist_2layer.txt");

	float accuracy = nn.accuracy_test(arr, labelvec);
	cout << "accuracy of simple test on training set: " << accuracy << endl;

	return accuracy;

}

float small_mnist_1layer_GD() {
	mat arr, labels;
	vec labelvec;
	from_files("input.txt", "output.txt", arr, labelvec);
	labelvec2mat(labels, labelvec);

	arr = arr / (255.0f);
	arr = arr - 0.5f;

	mat inputs = arr.submat(390, 0, 415, arr.n_cols - 1);

	vec empty;
	empty.reset();

	NeuralNet nn(26, 10, 1, empty, 3);
	int steps = 5000;
	cout << "training begins:" << endl;
	nn.train_GD(inputs, labels, steps, 0.1f, true, "small_mnist_1layer_GD.txt");

	float accuracy = nn.accuracy_test(inputs, labelvec);
	cout << "accuracy of simple test on training set: " << accuracy << endl;

	return accuracy;

}

float overnight_experiment2() {
	mat arr, labels;
	vec labelvec;
	from_files("input.txt", "output.txt", arr, labelvec);
	labelvec2mat(labels, labelvec);

	arr = arr / (255.0f);
	arr = arr - 0.5f;

	mat inputs = arr.submat(400, 0, 409, arr.n_cols - 1);

	vec one_hidden(1);
	one_hidden<<5;
	int steps;
	NeuralNet nn(10, 10, 2, one_hidden, 3);
	int max_time = 1000 * 60 ; // 5 hours
	steps = 10000;
	//cout << "training begins:" << endl;
	//vec weights = nn.get_weights();
//	nn.train_GD(inputs, labels, steps, 0.1f, true, "small_mnist_1layer_GD_time_test2_2.txt", true, max_time);
	nn.set_TRM_parameters(0.1, 0.7, 0.1, 2.0);
	float ballSize = 0.01f;
	max_time = max_time * 20;
	//nn.set_weights(weights);
	cout << "training begins:" << endl;
	nn.train_TRM(inputs, labels, steps, ballSize, true, "small_mnist_1layer_TRM_time_test2_2.txt", 0.0f, true, max_time);




	float accuracy = nn.accuracy_test(inputs, labelvec);
	cout << "accuracy of simple test on training set: " << accuracy << endl;

	return accuracy;

}



float overnight_experiment() {
	mat arr, labels;
	vec labelvec;
	from_files("input.txt", "output.txt", arr, labelvec);
	labelvec2mat(labels, labelvec);

	arr = arr / (255.0f);
	arr = arr - 0.5f;

	mat inputs = arr.submat(400, 0, 409, arr.n_cols - 1);

	vec empty;
	empty.reset();
	int steps;
	NeuralNet nn(10, 10, 1, empty, 3);
	int max_time = 1000 * 60 * 60 * 7; // 3 hours
	steps = 10000;
	cout << "training begins:" << endl;
	vec weights = nn.get_weights();
	nn.train_GD(inputs, labels, steps, 0.1f, true, "small_mnist_1layer_GD_time_test3.txt", true, max_time);
	nn.set_TRM_parameters(0.3, 0.7, 0.5, 2.0);
	float ballSize = 100.0f;
	nn.set_weights(weights);
	cout << "training begins:" << endl;
	nn.train_TRM(inputs, labels, steps, ballSize, true, "small_mnist_1layer_TRM_time_test3.txt",0.0f, true, max_time);




	float accuracy = nn.accuracy_test(inputs, labelvec);
	cout << "accuracy of simple test on training set: " << accuracy << endl;

	return accuracy;

}

float small_mnist_1layer_TRM() {
	mat arr, labels;
	vec labelvec;
	from_files("input.txt", "output.txt", arr, labelvec);
	labelvec2mat(labels, labelvec);

	arr = arr / (255.0f);
	arr = arr - 0.5f;

	mat inputs = arr.submat(400, 0, 409, arr.n_cols - 1);

	vec empty;
	empty.reset();
	int steps;
	NeuralNet nn(10, 10, 1, empty, 3);

	steps = 10000;
	cout << "training begins:" << endl;
	nn.train_GD(inputs, labels, steps, 0.1f, true, "small_mnist_1layer_GD_215_1.txt");
	
	//vec weights = nn.get_weights();
	nn.set_TRM_parameters(0.3, 0.7, 0.5, 2.0);


	float ballSize = 100.0f;
	
	steps = 50000;
	//cout << "training begins again:" << endl;
	//nn.train_GD(inputs, labels, steps, 0.1f, true, "small_mnist_1layer_GD_215_2.txt");

	//float error = nn.test(inputs, labels);
	

	//nn.set_weights(weights);
	cout << "training begins:" << endl;
	nn.train_TRM(inputs, labels, steps, ballSize, true, "small_mnist_1layer_TRM_215_1.txt");




	float accuracy = nn.accuracy_test(inputs, labelvec);
	cout << "accuracy of simple test on training set: " << accuracy << endl;

	return accuracy;

}

float simplest_mnist_1layer() {
	mat arr, labels;
	vec labelvec;
	from_files("input.txt", "output.txt", arr, labelvec);
	labelvec2mat(labels, labelvec);

	arr = arr / (255.0f);
	arr = arr - 0.5f;

	vec empty;
	empty.reset();

	NeuralNet nn(784, 10, 1,empty, 3);
	nn.set_TRM_parameters(0.1, 0.8, 0.7, 1.5);
	float ballSize = 0.01f;
	int steps = 100;
	cout << "training begins:" << endl;
	nn.train_TRM(arr, labels, steps, ballSize, true, "simples_mnist_TRM.txt");

	float accuracy = nn.accuracy_test(arr, labelvec);
	cout << "accuracy of simple test on training set: " << accuracy << endl;

	return accuracy;
}

float mnist_2layer_TRM() {
	mat arr, labels;
	vec labelvec;
	from_files("input.txt", "output.txt", arr, labelvec);
	labelvec2mat(labels, labelvec);

	arr = arr / (255.0f);
	arr = arr - 0.5f;

	vec ns(2);
	ns << 100 << 50 << endr;


	NeuralNet nn(784, 10, 2, ns, 4);
	nn.set_TRM_parameters(0.1, 0.8, 0.7, 1.5);
	float ballSize = 0.001f;
	int steps = 100;
	cout << "training begins:" << endl;
	nn.train_TRM(arr, labels, steps, ballSize, true, "mnist_2layer.txt");

	float accuracy = nn.accuracy_test(arr, labelvec);
	cout << "accuracy of simple test on training set: " << accuracy << endl;

	return accuracy;

}

float mnist_2layer() {
	mat arr, labels;
	vec labelvec;
	from_files("input.txt", "output.txt", arr, labelvec);
	labelvec2mat(labels, labelvec);

	arr = arr / (255.0f);
	arr = arr - 0.5f;

	vec ns(2);
	ns << 100 << 50 << endr;


	NeuralNet nn(784, 10, 2, ns, 4);
	nn.train_GD(arr, labels, 10000, 0.01f, true, "mnist_2layer.txt");

	float accuracy = nn.accuracy_test(arr, labelvec);
	cout << "accuracy of simple test on training set: " << accuracy << endl;

	return accuracy;

}

float simpletest() {

	mat arr, labels;
	vec labelvec;
	from_files("input.txt", "output.txt", arr, labelvec);
	labelvec2mat(labels, labelvec);

	arr = arr / (255.0f);
	arr = arr - 0.5f;

	vec ns(1);
	ns << 100 << endr;

	//preset_start("preset_start.txt", 784, 10, 2, ns);

	NeuralNet nn(784, 10, 2, ns, 3);
	//nn.train_GD(arr, labels, 1000, 0.0000000000000005f, true);
	nn.train_GD_Alr(arr, labels, 300, 0.01f, 2.0f, 0.01f, true, "");
	//nn.print_weights();
	//nn.print_grad();
	float accuracy = nn.accuracy_test(arr, labelvec);
	cout << "accuracy of simple test on training set: " << accuracy << endl;

	return accuracy;
}

void redo_txt(int num_examples, string filename_input, string filename_output) {
	mat arr, labels;
	vec labelvec;
	ReadMNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 60000, 784, arr, labelvec);
	save_file(filename_input, filename_output, arr, labelvec, num_examples);
}

void basic_test_GD() {
	mat inputs(2,10);
	mat outputs(1,10);

	inputs << 1 << 5 << 4 << 7 << 8 << 9 << 1 << 2 << 1 << 9 << endr
		<< 3 << 2 << 6 << 1 << 5 << 3 << 23 << 2 << 6 << 1 << endr;
	//inputs << 1 << 5 << endr
	//	<< 3 << 2 << endr;


	for (int i = 0; i < outputs.n_elem; i++) {
		// ie weights should be 1 and 1
		outputs(0,i) = inputs(0, i) + inputs(1, i);
	}

	mat w(1, 2);
	w << 0.5 << 1.5 << endr;
	vec empty;
	empty.reset();

	NeuralNet nn(2, 1, 1, empty, 1);
	nn.train_GD(inputs, outputs,200000,0.00001f,true, "GD_error.txt");
	//cout << "should have weights 1 1" << endl;
	mat results = nn.apply(inputs);
	bool passed = true;
	for (int i = 0; i < 10; i++) {
		if (abs(results(0, i) - outputs(0, i)) > 0.000001f) {
			passed = false;
			cerr << "error with system!!!!!!" << endl;
		}
	}
	if (passed) {
		cout << "PASSED basic test" << endl;
	}
}

void two_layer_test_TRM() {
	mat inputs(2, 10);
	mat outputs(1, 10);

	inputs << 1 << 5 << 4 << 7 << 8 << 9 << 1 << 2 << 1 << 9 << endr
		<< 3 << 2 << 6 << 1 << 5 << 3 << 23 << 2 << 6 << 1 << endr;
	for (int i = 0; i < outputs.n_elem; i++) {
		// ie weights should be 1 and 1
		outputs(0, i) = inputs(0, i) + inputs(1, i);
	}

	vec hidden(1);
	hidden << 2 << endr;


	// 3 for sigmoid
	NeuralNet nn(2, 1, 2, hidden,1);
	nn.set_TRM_parameters(0.1, 0.8, 0.1, 5);
	float ballSize = 0.1f;
	int steps = 1000;
	nn.train_TRM(inputs, outputs, steps, ballSize, true, "TRM_error.txt");
	nn.print_weights();
	//nn.print_weights();
	mat results = nn.apply(inputs);
	//results.print("results: ");
	//outputs.print("expected results: ");
	bool passed = true;
	for (int i = 0; i < 10; i++) {
		if (abs(results(0, i) - outputs(0, i)) > 0.0001f) {
			passed = false;
			cerr << "error with system!!!!!!" << endl;
		}
	}
	if (passed) {
		cout << "PASSED two_layer_test" << endl;
	}

}

void two_layer_test() {
	mat inputs(2, 10);
	mat outputs(1, 10);

	inputs << 1 << 5 << 4 << 7 << 8 << 9 << 1 << 2 << 1 << 9 << endr
		<< 3 << 2 << 6 << 1 << 5 << 3 << 23 << 2 << 6 << 1 << endr;
	for (int i = 0; i < outputs.n_elem; i++) {
		// ie weights should be 1 and 1
		outputs(0, i) = inputs(0, i) + inputs(1, i);
	}

	vec hidden(1);
	hidden << 2 << endr;



	NeuralNet nn(2, 1, 2, hidden);
	nn.train_GD(inputs, outputs, 1000, 1.0f, false, "");
	//nn.print_weights();
	mat results = nn.apply(inputs);
	//results.print("results: ");
	//outputs.print("expected results: ");
	bool passed = true;
	for (int i = 0; i < 10; i++) {
		if (abs(results(0, i) - outputs(0, i)) > 0.0001f) {
			passed = false;
			cerr << "error with system!!!!!!" << endl;
		}
	}
	if (passed) {
		cout << "PASSED two_layer_test" << endl;
	}

}

void two_layer_test_large_hidden() {
	mat inputs(2, 10);
	mat outputs(1, 10);

	inputs << 1 << 5 << 4 << 7 << 8 << 9 << 1 << 2 << 1 << 9 << endr
		<< 3 << 2 << 6 << 1 << 5 << 3 << 23 << 2 << 6 << 1 << endr;
	for (int i = 0; i < outputs.n_elem; i++) {
		// ie weights should be 1 and 1
		outputs(0, i) = inputs(0, i) + inputs(1, i);
	}

	vec hidden(1);
	hidden << 1000 << endr;


	NeuralNet nn(2, 1, 2, hidden);
	nn.train_GD(inputs, outputs, 1000,1.0f, false, "");

	mat results = nn.apply(inputs);
	//results.print("results: ");
	//outputs.print("expected results: ");
	bool passed = true;
	for (int i = 0; i < 10; i++) {
		if (abs(results(0, i) - outputs(0, i)) > 0.1f) {
			passed = false;
			cerr << "error with system!!!!!!" << endl;
		}
	}
	if (passed) {
		cout << "PASSED two_layer_test_large_hidden" << endl;
	}

}

void seven_layer_test_medium_hidden_TRM_cd() {
	mat inputs(2, 10);
	mat outputs(1, 10);

	inputs << 1 << 5 << 4 << 7 << 8 << 9 << 1 << 2 << 1 << 9 << endr
		<< 3 << 2 << 6 << 1 << 5 << 3 << 23 << 2 << 6 << 1 << endr;
	for (int i = 0; i < outputs.n_elem; i++) {
		// ie weights should be 1 and 1
		outputs(0, i) = inputs(0, i) + inputs(1, i);
	}

	vec hidden(6);
	hidden << 4 << 6 << 2 << 5 << 2 << 7 << endr;

	vec nlin(6);
	nlin << 3 << 3 << 3 << 3 << 3 << 3 << endr;

	NeuralNet nn(2, 1, 7, hidden, nlin);
	nn.set_TRM_parameters(0.2, 0.8, 0.5, 2.0);
	vec weights = nn.get_weights();
	int steps = 1000;
	nn.train_TRM_cd(inputs, outputs, steps, 1.0f, true, "7layerTRMcd.txt", 0.0f, true);
	nn.set_weights(weights);
	nn.train_GD(inputs, outputs, steps*10, 0.0001, true, "7layerGD.txt", true, 100000000000000);
	mat results = nn.apply(inputs);
	//results.print("results: ");
	//outputs.print("expected results: ");
	bool passed = true;
	for (int i = 0; i < 10; i++) {
		if (abs(results(0, i) - outputs(0, i)) > 0.1f) {
			passed = false;
			cerr << "error with system!!!!!!" << endl;
		}
	}
	if (passed) {
		cout << "PASSED seven_layer_test_medium_hidden" << endl;
	}

}

void seven_layer_test_medium_hidden() {
	mat inputs(2, 10);
	mat outputs(1, 10);

	inputs << 1 << 5 << 4 << 7 << 8 << 9 << 1 << 2 << 1 << 9 << endr
		<< 3 << 2 << 6 << 1 << 5 << 3 << 23 << 2 << 6 << 1 << endr;
	for (int i = 0; i < outputs.n_elem; i++) {
		// ie weights should be 1 and 1
		outputs(0, i) = inputs(0, i) + inputs(1, i);
	}

	vec hidden(6);
	hidden << 50 << 50 << 50 << 50 << 50 << 50 << endr;

	vec nlin(6);

	nlin << 3 << 3 << 3 << 3 << 3 << 3 << endr;

	NeuralNet nn(2, 1, 7, hidden, nlin);
	nn.train_GD(inputs, outputs, 100000, 0.00001f, true, "7layerGD.txt");

	mat results = nn.apply(inputs);
	//results.print("results: ");
	//outputs.print("expected results: ");
	bool passed = true;
	for (int i = 0; i < 10; i++) {
		if (abs(results(0, i) - outputs(0, i)) > 0.1f) {
			passed = false;
			cerr << "error with system!!!!!!" << endl;
		}
	}
	if (passed) {
		cout << "PASSED seven_layer_test_medium_hidden" << endl;
	}

}

void basic_sigmoid_test() {
	mat inputs(2, 10);
	mat outputs(1, 10);


	inputs << 1 << 5 << 4 << 7 << 8 << 9 << 1 << 2 << 1 << 9 << endr
		<< 3 << 2 << 6 << 1 << 5 << 3 << 23 << 2 << 6 << 1 << endr;

	//inputs << 1 << 5 << endr
	//	<< 3 << 2 << endr;

	float temp;
	for (int i = 0; i < outputs.n_elem; i++) {
		// ie weights should be 1 and 1
		temp = inputs(0, i) - inputs(1, i);
		if (temp > 0) {
			outputs(0, i) = 1;
		}
		else {
			outputs(0, i) = 0;
		}
	}

	mat w(1, 2);
	w << 1 << -1 << endr;
	vec empty;
	empty.reset();


	NeuralNet nn(2, 1, 1, empty, 3, &w);
	//cout << "initial weights: " << endl;
	//nn.print_weights();
	nn.train_GD(inputs, outputs, 100000, 0.1f, true, "GD.txt");
	//cout << "weights" << endl;
	//nn.print_weights();
	//cout << "grad: " << endl;
	//nn.print_grad();

	mat results = nn.apply(inputs);
	bool passed = true;
	for (int i = 0; i < 10; i++) {
		if (abs(results(0, i) - outputs(0, i)) > 0.1f) {
			
			passed = false;
			cerr << "error with system!!!!!!" << endl;
		}
	}
	if (passed) {
		cout << "PASSED basic sigmoid test" << endl;
	}
}

void redo_io_files(int n_examples) {
	mat arr;
	vec labels;
	ReadMNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 60000, 784, arr, labels);
	save_file("input.txt", "output.txt", arr, labels, n_examples);
}

void basic_test_hv() {
	mat inputs(4, 1);
	mat outputs(2,1);

	inputs << 1 << endr << 2 << endr << 2 << endr << 1 << endr;
	outputs << 2 << endr << 4 << endr;

	mat w(2, 4);
	w << 1 << -1 << 1 << -1 << endr << 1 << 3 << 1 << -2 << endr;
	vec empty;
	empty.reset();

	vec v(8);
	v << 1 << 2 << 0.5 << -1 << -0.5 << 1 << -1 << 3 << endr;


	NeuralNet nn(4,2,1);
	nn.forwback(inputs, outputs);
	vec hv = nn.Hv(v);
	//cout << "should have weights 1 1" << endl;

	vec ground_truth(8);
	ground_truth << 0 << 5 << 0 << 10 << 0 << 10 << 0 << 5 << endr;
	bool passed = true;
	for (int i = 0; i < 8; i++) {
		if (abs(hv(i) - ground_truth(i)) > 0.00001) {
			passed = false;
		}
	}
	if (passed) {
		cout << "PASSED basic_test_hv" << endl;
	}
	else {
		cout << "FAILED basic_test_hv" << endl;
	}
}

void basic_test_GD2() {
	mat inputs(2, 10);
	mat outputs(1, 10);

	inputs << 1 << 5 << 4 << 7 << 8 << 9 << 1 << 2 << 1 << 9 << endr
		<< 3 << 2 << 6 << 1 << 5 << 3 << 23 << 2 << 6 << 1 << endr;
	//inputs << 1 << 5 << endr
	//	<< 3 << 2 << endr;


	for (int i = 0; i < outputs.n_elem; i++) {
		// ie weights should be 1 and 1
		outputs(0, i) = inputs(0, i) + inputs(1, i);
	}

	//mat w(1, 2);
	//w << 0.5 << 1.5 << endr;
	vec empty;
	empty.reset();



	NeuralNet nn(2, 1, 2);
	nn.set_TRM_parameters(0.25, 0.75, 0.9, 1.1);
	float ballSize = 0.1f;
	int steps = 1000;
	nn.train_GD(inputs, outputs, steps, 0.01f, true, "GD_error2.txt");
	nn.print_weights();
	//cout << "should have weights 1 1" << endl;
	mat results = nn.apply(inputs);
	bool passed = true;
	for (int i = 0; i < 10; i++) {
		if (abs(results(0, i) - outputs(0, i)) > 0.000001f) {
			passed = false;
			cerr << "error with system!!!!!!" << endl;
		}
	}
	if (passed) {
		cout << "PASSED basic test" << endl;
	}
}

void basic_test_TRM_cd() {
	mat inputs(2, 10);
	mat outputs(1, 10);

	inputs << 1 << 5 << 4 << 7 << 8 << 9 << 1 << 2 << 1 << 9 << endr
		<< 3 << 2 << 6 << 1 << 5 << 3 << 23 << 2 << 6 << 1 << endr;
	//inputs << 1 << 5 << endr
	//	<< 3 << 2 << endr;


	for (int i = 0; i < outputs.n_elem; i++) {
		// ie weights should be 1 and 1
		outputs(0, i) = inputs(0, i) + inputs(1, i);
	}

	//mat w(1, 2);
	//w << 0.5 << 1.5 << endr;
	vec empty;
	empty.reset();



	NeuralNet nn(2, 1, 2);
	nn.set_TRM_parameters(0.25, 0.75, 0.5, 2.0);
	float ballSize = 0.1f;
	int steps = 1000;
	nn.train_TRM_cd(inputs, outputs, steps, ballSize, true, "TRM_cd_error.txt");
	cout << "finished cd" << endl;
	nn.print_weights();
	//cout << "should have weights 1 1" << endl;
	mat results = nn.apply(inputs);
	bool passed = true;
	for (int i = 0; i < 10; i++) {
		if (abs(results(0, i) - outputs(0, i)) > 0.000001f) {
			passed = false;
			cerr << "error with system!!!!!!" << endl;
		}
	}
	if (passed) {
		cout << "PASSED basic test" << endl;
	}

}

void basic_test_TRM2() {
	mat inputs(2, 10);
	mat outputs(1, 10);

	inputs << 1 << 5 << 4 << 7 << 8 << 9 << 1 << 2 << 1 << 9 << endr
		<< 3 << 2 << 6 << 1 << 5 << 3 << 23 << 2 << 6 << 1 << endr;
	//inputs << 1 << 5 << endr
	//	<< 3 << 2 << endr;


	for (int i = 0; i < outputs.n_elem; i++) {
		// ie weights should be 1 and 1
		outputs(0, i) = inputs(0, i) + inputs(1, i);
	}

	//mat w(1, 2);
	//w << 0.5 << 1.5 << endr;
	vec empty;
	empty.reset();



	NeuralNet nn(2, 1, 2);
	nn.set_TRM_parameters(0.25, 0.75, 0.9, 1.1);
	float ballSize = 0.1f;
	int steps = 100000;
	nn.train_TRM(inputs, outputs, steps, ballSize, true, "TRM_error2.txt");
	nn.print_weights();
	//cout << "should have weights 1 1" << endl;
	mat results = nn.apply(inputs);
	bool passed = true;
	for (int i = 0; i < 10; i++) {
		if (abs(results(0, i) - outputs(0, i)) > 0.000001f) {
			passed = false;
			cerr << "error with system!!!!!!" << endl;
		}
	}
	if (passed) {
		cout << "PASSED basic test" << endl;
	}

}

void basic_test_TRM() {
	mat inputs(2, 10);
	mat outputs(1, 10);

	inputs << 1 << 5 << 4 << 7 << 8 << 9 << 1 << 2 << 1 << 9 << endr
		<< 3 << 2 << 6 << 1 << 5 << 3 << 23 << 2 << 6 << 1 << endr;
	//inputs << 1 << 5 << endr
	//	<< 3 << 2 << endr;


	for (int i = 0; i < outputs.n_elem; i++) {
		// ie weights should be 1 and 1
		outputs(0, i) = inputs(0, i) + inputs(1, i);
	}

	mat w(1, 2);
	w << 0.5 << 1.5 << endr;
	vec empty;
	empty.reset();



	NeuralNet nn(2, 1, 1, empty,1);
	int steps = 100000;
	nn.train_GD(inputs, outputs, steps, 0.00000001f, true, "");

	//nn.set_TRM_parameters(0.25, 0.75, 0.5, 1.5);
	float ballSize = 0.1f;
	steps = 100;
	nn.train_TRM(inputs, outputs, steps, ballSize, true, "TRM_error.txt");
	nn.print_weights();
	//cout << "should have weights 1 1" << endl;
	mat results = nn.apply(inputs);
	bool passed = true;
	for (int i = 0; i < 10; i++) {
		if (abs(results(0, i) - outputs(0, i)) > 0.000001f) {
			passed = false;
			cerr << "error with system!!!!!!" << endl;
		}
	}
	if (passed) {
		cout << "PASSED basic test" << endl;
	}

}

void basic_sigmoid_test_TRM() {
	mat inputs(2, 10);
	mat outputs(1, 10);


	inputs << 1 << 5 << 4 << 7 << 8 << 9 << 1 << 2 << 1 << 9 << endr
		<< 3 << 2 << 6 << 1 << 5 << 3 << 23 << 2 << 6 << 1 << endr;

	//inputs << 1 << 5 << endr
	//	<< 3 << 2 << endr;

	float temp;
	for (int i = 0; i < outputs.n_elem; i++) {
		// ie weights should be 1 and 1
		temp = inputs(0, i) - inputs(1, i);
		if (temp > 0) {
			outputs(0, i) = 1;
		}
		else {
			outputs(0, i) = 0;
		}
	}

	mat w(1, 2);
	w << 1 << -1 << endr;
	vec empty;
	empty.reset();


	NeuralNet nn(2, 1, 1, empty, 3, &w);
	//cout << "initial weights: " << endl;
	//nn.print_weights();
	nn.train_GD(inputs, outputs, 10000, 0.1f, true, "GD.txt");
	nn.set_TRM_parameters(0.25, 0.75, 0.5, 2.0);
	float ballSize = 0.1f;
	cout << " " << endl;
	cout << " " << endl;
	cout << " " << endl;
	int steps = 15;
	nn.train_TRM(inputs, outputs, steps, ballSize, true, "TRM_error.txt");
	//cout << "weights" << endl;
	//nn.print_weights();
	//cout << "grad: " << endl;
	//nn.print_grad();

	mat results = nn.apply(inputs);
	bool passed = true;
	for (int i = 0; i < 10; i++) {
		if (abs(results(0, i) - outputs(0, i)) > 0.1f) {

			passed = false;
			cerr << "error with system!!!!!!" << endl;
		}
	}
	if (passed) {
		cout << "PASSED basic sigmoid test" << endl;
	}
}

void test_cg_TRM() {
	mat inputs(2, 10);
	mat outputs(1, 10);

	inputs << 1 << 5 << 4 << 7 << 8 << 9 << 1 << 2 << 1 << 9 << endr
		<< 3 << 2 << 6 << 1 << 5 << 3 << 23 << 2 << 6 << 1 << endr;

	for (int i = 0; i < outputs.n_elem; i++) {
		outputs(0, i) = inputs(0, i) + inputs(1, i);
	}

	vec empty;
	empty.reset();
	NeuralNet nn(2, 1, 2, 1);
	nn.set_TRM_parameters(0.25, 0.75, 0.5, 1.5);
	float ballSize = 0.1f;
	int steps = 2000;
	nn.train_TRM(inputs, outputs, steps, ballSize, true, "TRM_error.txt");
	vec p1(12), g(12);
	nn.get_p1_TRM(p1, g);
	vec p0 = nn.cg(g);

}

void get_p1_test_TRM() {
	mat inputs(2, 10);
	mat outputs(1, 10);

	inputs << 1 << 5 << 4 << 7 << 8 << 9 << 1 << 2 << 1 << 9 << endr
		<< 3 << 2 << 6 << 1 << 5 << 3 << 23 << 2 << 6 << 1 << endr;

	for (int i = 0; i < outputs.n_elem; i++) {
		outputs(0, i) = inputs(0, i) + inputs(1, i);
	}

	vec empty;
	empty.reset();
	NeuralNet nn(2, 1, 2, 4);
	nn.set_TRM_parameters(0.25, 0.75, 0.5, 1.5);
	nn.forwback(inputs, outputs);
	vec p1(30), g(30);
	nn.get_p1_TRM(p1, g);

}

void runtests() {
	basic_test_GD();
	two_layer_test();
	two_layer_test_large_hidden();
	seven_layer_test_medium_hidden();
	basic_sigmoid_test();
	basic_test_hv();
}

void timing_test() {
	mat arr, labels;
	vec labelvec;
	from_files("input.txt", "output.txt", arr, labelvec);
	labelvec2mat(labels, labelvec);

	arr = arr / (255.0f);
	arr = arr;

	vec empty;
	empty.reset();

	int n_weights = 7840;
	vec v(n_weights);
	v.randu();

	NeuralNet nn(784, 10, 1, empty, 4);
	nn.set_TRM_parameters(0.1, 0.8, 0.7, 1.5);
	nn.forwback(arr, labels);
	
	float sum = 0;
	int total = 100;
	clock_t t;
	t = clock();
	

	for (int i = 0; i < total; i++) {
		t = clock();
		v = nn.Hv(v);
		sum = sum + (clock() - t);
	}

	cout << "average time = " << sum / float(total) << " ms" << endl;
}

int main()
{


	//overnight_experiment2();
	//small_mnist_1layer_TRM();
	//test_cg_TRM();
	//basic_test_TRM_cd();
	//redo_io_files(300);
	//runtests();
	seven_layer_test_medium_hidden_TRM_cd();
	//simplest_mnist_1layer();
//	float accuracy = mnist_2layer();
	//simplest_mnist_1layer();
	//clock_t t;
	//t = clock();
	//system("pause");
	//basic_test_TRM();
	//basic_test_GD();
	//basic_test_GD2();
	//t = clock() - t;
	//two_layer_test_TRM();
	//two_layer_test();
	cout << "should have printed to files... " << endl;
	std::cout << "Hello World!" << std::endl;
	system("pause");
	return 0;
}

