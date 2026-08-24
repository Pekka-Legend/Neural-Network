#pragma once

#include <vector>
#include "layer.h"
#include "neuron.h"
#include "matrix.h"

const double learningRate = .001;

double loss(double prediction, double target)
{
	return pow(prediction - target, 2);
}

double lossDerivative(double prediction, double target)
{
	return 2 * (prediction - target);
}

double generateTestTarget(double input)
{
	return sin(input);
}

Vector generateOutputLayerErrors(bool useRelu, Vector preActivationValues, Vector postActivationValues, Vector target)
{
	std::vector<double> errorList;
	std::vector<double> postValuesList = postActivationValues.getVector();
	std::vector<double> preValuesList = preActivationValues.getVector();
	std::vector<double> targetList = target.getVector();
	errorList.resize(postActivationValues.size());
	for (int i = 0; i < errorList.size(); i++)
	{
		double activationDerivative = 1.0;
		if (useRelu)
		{
			activationDerivative = leakyReluDerivative(preValuesList[i]);
		}
		errorList[i] = lossDerivative(postValuesList[i], targetList[i]) * activationDerivative;
	}
	return Vector(errorList);
}

class Network
{
private:
	std::vector<Layer> layers;
public:
	Network()
	{

	}
	Network(std::vector<Layer> newLayers)
	{
		layers = newLayers;
	}
	std::vector<Layer> getLayers()
	{
		return layers;
	}
	void setLayers(std::vector<Layer> newLayers)
	{
		layers = newLayers;
	}
	void printNetwork()
	{
		for (int i = 0; i < layers.size(); i++)
		{
			std::cout << "Layer " << (i + 1) << ":" << std::endl;
			layers[i].print();
			std::cout << "\n";
		}
	}
	void train(Vector input, Vector target)
	{
		Vector passedVector = Vector(input.getVector());
		std::vector<Vector> inputCache;
		inputCache.clear();
		for (int i = 0; i < layers.size(); i++)
		{
			inputCache.push_back(passedVector);
			passedVector = layers[i].forward(passedVector);
		}
		Vector postOutputValues = Vector(layers[layers.size() - 1].getPostActivationCache());
		Vector preOutputValues = Vector(layers[layers.size() - 1].getPreActivationCache());

		Vector currentError = generateOutputLayerErrors(layers[layers.size() - 1].getUseRelu(), preOutputValues, postOutputValues, target);

		std::vector<Matrix> weightGradients(layers.size());
		std::vector<Vector> biasGradients(layers.size());

		for (int i = layers.size() - 1; i >= 0; i--)
		{
			if (i < layers.size() - 1) currentError = layers[i].backward(currentError, layers[i+1].getWeightMatrix());
			weightGradients[i] = MultiplyMatrix(Transpose(Vector2Matrix(currentError)), Vector2Matrix(inputCache[i]));
			biasGradients[i] = currentError;

			
		}

		for (int i = 0; i < layers.size(); i++)
		{
			layers[i].learn(weightGradients[i], biasGradients[i], learningRate);
		}
		//printNetwork();
	}
};