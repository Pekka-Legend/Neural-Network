#pragma once
#include <vector>

class Vector
{
private:
	std::vector<double> vect;
public:
	Vector()
	{

	}
	Vector(std::vector<double> inVect)
	{
		vect = inVect;
	}
	void setVector(std::vector<double> newVector)
	{
		vect = newVector;
	}
	std::vector<double> getVector()
	{
		return vect;
	}
	int size()
	{
		return vect.size();
	}
};

class Matrix
{
private:
	std::vector<std::vector<double>> matrix;
public:
	Matrix()
	{

	}
	Matrix(std::vector<std::vector<double>> newMat)
	{
		matrix = newMat;
	}
	void setMatrix(std::vector<std::vector<double>> newMatrix)
	{
		matrix = newMatrix;
	}
	std::vector<std::vector<double>> getMatrix()
	{
		return matrix;
	}
	Vector getRow(int index)
	{
		return Vector(matrix[index]);
	}
	void printMatrix()
	{
		std::cout << matrix.size() << "x" << matrix[0].size() << " Matrix: " <<std::endl;
		for (int i = 0; i < matrix.size(); i++)
		{
			for (int j = 0; j < matrix[i].size(); j++)
			{
				std::cout << " " << matrix[i][j] << " ";
			}
			std::cout << std::endl;
		}
	}
};

Matrix MultiplyMatrix(Matrix m1, Matrix m2)
{
	Matrix output;
	std::vector<std::vector<double>> outMat;
	std::vector<std::vector<double>> m1Mat = m1.getMatrix();
	std::vector<std::vector<double>> m2Mat = m2.getMatrix();
	if (m1Mat.size() > 0 && m2Mat.size() > 0 && m1Mat[0].size() == m2Mat.size())
	{
		outMat.resize(m1Mat.size(), std::vector<double>(m2Mat[0].size()));
		for (int i = 0; i < outMat.size(); i++)
		{
			for (int j = 0; j < outMat[i].size(); j++)
			{
				double sum = 0;
				for (int k = 0; k < m1Mat[0].size(); k++)
				{
					sum += m1Mat[i][k] * m2Mat[k][j];
				}
				outMat[i][j] = sum;
			}
		}
		output.setMatrix(outMat);
		//output.printMatrix();
	}
	else
	{
		std::cout << "Multiply Operation Failed" << std::endl;
	}
	return output;
}

Matrix Vector2Matrix(Vector vect)
{
	Matrix output;
	std::vector<std::vector<double>> outMat;
	outMat.push_back(vect.getVector());
	output.setMatrix(outMat);
	return output;
}

Matrix Transpose(Matrix matrix)
{
	//matrix.printMatrix();
	Matrix transpose;
	std::vector<std::vector<double>> matrixList = matrix.getMatrix();
	std::vector<std::vector<double>> newMat;

	newMat.resize(matrixList[0].size(), std::vector<double>(matrixList.size()));
	for (int i = 0; i < newMat.size(); i++)
	{
		for (int j = 0; j < newMat[0].size(); j++)
		{
			newMat[i][j] = matrixList[j][i];
		}
	}

	transpose.setMatrix(newMat);
	//transpose.printMatrix();
	return transpose;
}

Matrix HadamardProduct(Matrix m1, Matrix m2)
{
	Matrix output;
	std::vector<std::vector<double>> outMat;
	std::vector<std::vector<double>> m1Mat = m1.getMatrix();
	std::vector<std::vector<double>> m2Mat = m2.getMatrix();
	if (m1Mat.size() > 0 && m2Mat.size() > 0 && m1Mat.size() == m2Mat.size() && m1Mat[0].size() == m2Mat[0].size())
	{
		outMat.resize(m1Mat.size(), std::vector<double>(m1Mat[0].size()));
		for (int i = 0; i < m1Mat.size(); i++)
		{
			for (int j = 0; j < m1Mat[0].size(); j++)
			{
				outMat[i][j] = m1Mat[i][j] * m2Mat[i][j];
			}
		}
		output.setMatrix(outMat);
	}
	else
	{
		std::cout << "Operation Failed" << std::endl;
	}

	return output;
}