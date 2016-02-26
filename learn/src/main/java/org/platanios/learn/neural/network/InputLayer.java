package org.platanios.learn.neural.network;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
class InputLayer extends Layer {
    private final VectorVariable inputVariable;
    private final Variable outputVariable;

    InputLayer(VariablesManager variablesManager, int size) {
        super(variablesManager, size);
        inputVariable = variablesManager.vectorVariable(size);
        outputVariable = variablesManager.layerVariable(this);
    }

    InputLayer(VariablesManager variablesManager, String variableName, int size) {
        super(variablesManager, size);
        inputVariable = variablesManager.vectorVariable(variableName, size);
        outputVariable = variablesManager.layerVariable(this);
    }

    @Override
    Layer[] inputLayers() {
        return new Layer[0];
    }

    @Override
    Variable[] inputVariables() {
        return new Variable[] { inputVariable };
    }

    @Override
    Variable outputVariable() {
        return outputVariable;
    }

    @Override
    Vector computeValue(NetworkState state) {
        return state.get(inputVariable);
    }

    @Override
    Matrix recursiveGradient(NetworkState state, Variable variable) {
        if (inputVariable.equals(variable) || outputVariable.equals(variable))
            return Matrix.identity(inputVariable.size());
        else
            return Matrix.zeros(inputVariable.size(), variable.size());
    }

    @Override
    Matrix localGradient(NetworkState state, Variable variable) {
        if (variable.equals(outputVariable) || variable.equals(inputVariable))
            return Matrix.identity(variable.size());
        else
            return Matrix.zeros(outputSize, variable.size());
    }
}
