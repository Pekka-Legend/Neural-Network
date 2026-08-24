#pragma once

#include "neuron.h"
#include <vector>

class Layer
{
private:
	std::vector<Neuron> neurons;
	std::vector<double> preActivationCache;
	std::vector<double> postActivationCache;
	bool useRelu = false;

public:
	std::vector<Neuron>& getNeurons()
	{
		return neurons;
	}
	void addNeuron(Neuron n)
	{
		neurons.push_back(n);
	}
	void removeNeuron(int index)
	{
		neurons.erase(neurons.begin() + index);
	}
	void setNeuronsList(std::vector<Neuron> newNeurons)
	{
		neurons = newNeurons;
	}
	void fillWithEmptyNeurons(int count, int weightCount)
	{
		neurons.clear();
		neurons.resize(count);
		for (int i = 0; i < count; i++)
		{
			neurons[i] = Neuron(weightCount);
		}
	}
	Vector forward(Vector input)
	{
		preActivationCache.clear();
		postActivationCache.clear();
		std::vector<double> outputVect;
		outputVect.resize(neurons.size());
		for (int i = 0; i < neurons.size(); i++)
		{
			double preActivationValue = neurons[i].linearForward(input);
			double postActivationValue = preActivationValue;

			if (useRelu) postActivationValue = leakyRelu(preActivationValue);

			outputVect[i] = postActivationValue;

			preActivationCache.push_back(preActivationValue);
			postActivationCache.push_back(postActivationValue);
		}

		return Vector(outputVect);
	}
	Vector backward(Vector nextErrorSignal, Matrix nextLayerWeights)
	{
		Matrix weightError = MultiplyMatrix(Vector2Matrix(nextErrorSignal), nextLayerWeights);

		std::vector<double> activationDerivative;
		for (int i = 0; i < preActivationCache.size(); i++)
		{
			if (useRelu)
			{
				activationDerivative.push_back(leakyReluDerivative(preActivationCache[i]));
			}
			else
			{
				activationDerivative.push_back(1);//derivative of linear activation is 1
			}
		}

		Matrix derivativeMat = Vector2Matrix(Vector(activationDerivative));

		Matrix finalMat = HadamardProduct(weightError, derivativeMat);
		return Vector(finalMat.getMatrix()[0]);
	}
	void setUseRelu(bool in)
	{
		useRelu = in;
	}
	bool getUseRelu()
	{
		return useRelu;
	}
	void print()
	{
		for (int i = 0; i < neurons.size(); i++)
		{
			std::cout << "Neuron " << (i + 1) << ":" <<std::endl;
			neurons[i].print();
			std::cout << "\n";
		}
	}
	Vector getPreActivationCache()
	{
		return Vector(preActivationCache);
	}
	Vector getPostActivationCache()
	{
		return Vector(postActivationCache);
	}
	Matrix getWeightMatrix()
	{
		std::vector<std::vector<double>> weightMatrix;
		weightMatrix.resize(neurons.size(), std::vector<double>(neurons[0].getWeights().size()));

		for (int i = 0; i < weightMatrix.size(); i++)
		{
			for (int j = 0; j < weightMatrix[0].size(); j++)
			{
				weightMatrix[i][j] = neurons[i].getWeights().getVector()[j];
			}
		}

		return Matrix(weightMatrix);
	}

	void learn(Matrix dLdWl, Vector dLdbl, double learningRate)
	{
		for (int i = 0; i < neurons.size(); i++)
		{
			neurons[i].learn(dLdWl.getRow(i), dLdbl.getVector()[i], learningRate);
		}
	}

	void setNeuronWeightsFromMatrix(Matrix m)
	{
		std::vector<std::vector<double>> matList = m.getMatrix();
		for (int i = 0; i < matList.size(); i++)
		{
			neurons[i].setWeights(matList[i]);
		}
	}
	void setNeuronBiasesFromMatrix(Vector biasVect)
	{
		for (int i = 0; i < biasVect.size(); i++)
		{
			neurons[i].setB(biasVect.getVector()[i]);
		}
	}
};//finish implementing the matrix to neuron stuff, implement the dL/dWl and dL/bl thing for all layers
