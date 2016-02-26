package org.platanios.learn.neural.network;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
class ConstantLayer extends Layer {
    private final Vector value;
    private final Variable outputVariable;

    ConstantLayer(VariablesManager variablesManager, Vector value) {
        super(variablesManager, value.size());
        this.value = value;
        outputVariable = variablesManager.constantVariable(value);
    }

    @Override
    Layer[] inputLayers() {
        return new Layer[0];
    }

    @Override
    Variable[] inputVariables() {
        return new Variable[0];
    }

    @Override
    Variable outputVariable() {
        return outputVariable;
    }

    @Override
    Vector computeValue(NetworkState state) {
        return value;
    }

    @Override
    Matrix recursiveGradient(NetworkState state, Variable variable) {
        return Matrix.zeros(value.size(), variable.size());
    }

    @Override
    Matrix localGradient(NetworkState state, Variable variable) {
        if (variable.equals(outputVariable))
            return Matrix.identity(outputSize);
        return Matrix.zeros(value.size(), variable.size());
    }
}
