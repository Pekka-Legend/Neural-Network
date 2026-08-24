#include <iostream>
#include <vector>


#include <cstdlib>
#include <random>

#include "matrix.h"
#include "neuron.h"
#include "layer.h"
#include "network.h"

int main()
{
	std::srand(static_cast<unsigned int>(std::time(nullptr)));

	Matrix m1;
	Matrix m2;
	std::vector<std::vector<double>> mat1 = { {1, 2, 3}, {5, 6, 7} };
	std::vector<std::vector<double>> mat2 = { {4, 3}, { 1, 4 }, {2, 5} };

	m1.setMatrix(mat1);
	m2.setMatrix(mat2);
	MultiplyMatrix(m1, m2);

	Network network;
	Layer layer1;
	Layer layer2;

	layer1.fillWithEmptyNeurons(64, 1);
	layer1.setUseRelu(true);
	layer2.fillWithEmptyNeurons(1, 64);
	layer2.setUseRelu(false);
	network.setLayers({ layer1, layer2 });

	Vector input = Vector({ 1 });
	bool going = true;

	int epoch = 1;
	while (epoch < 200000)
	{
		network.train(input, Vector({generateTestTarget(input.getVector()[0])}));
		epoch++;
		input.setVector({ ((double)rand() / RAND_MAX * 20.0) - 10.0 });
	}

	std::cout << "--- Testing Curve ---" << std::endl;
	for (double testNumber = -5.0; testNumber <= 5.0; testNumber += 1.0)
	{
		Vector currentData = Vector({ testNumber });
		for (int i = 0; i < network.getLayers().size(); i++)
		{
			currentData = network.getLayers()[i].forward(currentData);
		}
		std::cout << "Input: " << testNumber
			<< " | Expected: " << generateTestTarget(testNumber)
			<< " | Prediction: " << currentData.getVector()[0] << std::endl;
	}

	return 0;
}