#pragma once
#include <vector>
#include "matrix.h"

double leakyRelu(double input)
{
	double output = std::max((double)0, input);
	if (output == 0) return .01 * input;
	return output;
}
double leakyReluDerivative(double x)
{
	if (x > 0)
		return 1;
	return 0.01;
}

class Neuron
{
private:
	Vector weights = Vector({ 0 });
	double b = .1;
public:
	Neuron()
	{

	}
	Neuron(int count)
	{
		std::vector<double> initialWeights;
		initialWeights.resize(count);
		for (int i = 0; i < count; i++) {
			// Keep weights between -1 and 1
			initialWeights[i] = (((double)rand() / RAND_MAX) * 2.0 - 1.0);
		}
		weights = Vector(initialWeights);

		// Force the bend to happen randomly between -5 and 5
		if (count == 1) // Only do this for the hidden layer
		{
			double hinge_x = (((double)rand() / RAND_MAX) * 10.0 - 5.0);
			b = -initialWeights[0] * hinge_x;
		}
		else // Standard small bias for the output layer
		{
			b = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * 0.1;
		}
	}
	Vector getWeights()
	{
		return weights;
	}
	double getB()
	{
		return b;
	}
	void setWeights(Vector Weights)
	{
		weights = Vector(Weights);
	}
	void setB(double B)
	{
		b = B;
	}
	double linearForward(Vector input)
	{
		Matrix mat = MultiplyMatrix(Vector2Matrix(weights), Transpose(Vector2Matrix(input)));
		return mat.getMatrix()[0][0] + b;
	}
	
	void learn(Vector dldW, double dldb, double learningRate)
	{
		std::vector<double> newWeights = weights.getVector();
		std::vector<double> dldWVect = dldW.getVector();
		for (int i = 0; i < weights.size(); i++)
		{
			newWeights[i] = newWeights[i] - learningRate * dldWVect[i];
		}
		weights = Vector(newWeights);
		b = b - learningRate * dldb;
	}
	void print()
	{
		std::vector<double> weightList = weights.getVector();
		for (int i = 0; i < weightList.size(); i++)
		{
			std::cout << "W" << (i+1) << ": " << weightList[i] << std::endl;
		}
		
		
		std::cout << "B: " << b << std::endl;
	}
};