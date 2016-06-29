package makina.learn.neural.network;

import makina.math.matrix.Matrix;
import makina.math.matrix.Vector;
import makina.math.matrix.Vectors;

/**
 * @author Emmanouil Antonios Platanios
 */
class FullyConnectedLayer extends SingleInputLayer {
    private final int numberOfHiddenUnits;
    private final boolean includeBiasTerm;
    private final MatrixVariable weights;
    private final VectorVariable bias;

    FullyConnectedLayer(VariablesManager variablesManager, Layer inputLayer, int numberOfHiddenUnits) {
        this(variablesManager, inputLayer, numberOfHiddenUnits, true);
    }

    FullyConnectedLayer(VariablesManager variablesManager, Layer inputLayer, int numberOfHiddenUnits, boolean includeBiasTerm) {
        super(variablesManager, inputLayer, numberOfHiddenUnits);
        this.numberOfHiddenUnits = numberOfHiddenUnits;
        this.includeBiasTerm = includeBiasTerm;
        this.weights = variablesManager.matrixVariable(numberOfHiddenUnits, inputVariable.size());
        if (includeBiasTerm)
            this.bias = variablesManager.vectorVariable(numberOfHiddenUnits);
        else
            this.bias = null;
    }

    FullyConnectedLayer(VariablesManager variablesManager, Layer inputLayer, int numberOfHiddenUnits, String weightsVariableName) {
        this(variablesManager, inputLayer, numberOfHiddenUnits, weightsVariableName, null);
    }

    FullyConnectedLayer(VariablesManager variablesManager,
                        Layer inputLayer,
                        int numberOfHiddenUnits,
                        String weightsVariableName,
                        String biasVariableName) {
        super(variablesManager, inputLayer, numberOfHiddenUnits);
        this.numberOfHiddenUnits = numberOfHiddenUnits;
        this.includeBiasTerm = biasVariableName != null;
        this.weights = variablesManager.matrixVariable(weightsVariableName, numberOfHiddenUnits, inputVariable.size());
        if (includeBiasTerm)
            this.bias = variablesManager.vectorVariable(biasVariableName, numberOfHiddenUnits);
        else
            this.bias = null;
    }

    @Override
    Variable[] parameters() {
        if (includeBiasTerm)
            return new Variable[] { weights, bias };
        else
            return new Variable[] { weights };
    }

    @Override
    Vector computeValue(NetworkState state) {
        Vector inputValue = inputLayer.value(state);
        Vector outputValue = Vectors.build(numberOfHiddenUnits, inputValue.type());
        Vector weights = state.get(this.weights);
        Vector bias = null;
        if (includeBiasTerm)
            bias = state.get(this.bias);
        for (int i = 0; i < numberOfHiddenUnits; i++) {
            double sum = 0.0;
            if (includeBiasTerm)
                sum += bias.get(i);
            for (int j = 0; j < inputVariable.size(); j++)
                sum += weights.get(i + j * numberOfHiddenUnits) * inputValue.get(j);
            outputValue.set(i, sum);
        }
        return outputValue;
    }

    @Override
    Matrix localGradient(NetworkState state, Variable variable) {
        if (variable.equals(outputVariable)) {
            return Matrix.identity(outputSize);
        } else if (variable.equals(inputVariable)) {
            return new Matrix(weights.value(state), outputSize);
        } else if (variable.equals(weights)) {
            Vector inputValue = inputLayer.value(state);
            Matrix gradient = Matrix.zeros(numberOfHiddenUnits, weights.size);
            for (int i = 0; i < numberOfHiddenUnits; i++)
                for (int j = 0; j < inputVariable.size(); j++)
                    gradient.setElement(i, i + j * numberOfHiddenUnits, inputValue.get(j));
            return gradient;
        } else if (includeBiasTerm && variable.equals(bias)) {
            return Matrix.identity(numberOfHiddenUnits);
        } else {
            return Matrix.zeros(numberOfHiddenUnits, variable.size);
        }
    }
}
