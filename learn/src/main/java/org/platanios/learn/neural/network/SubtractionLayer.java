package org.platanios.learn.neural.network;

import org.platanios.math.matrix.Matrix;
import org.platanios.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
class SubtractionLayer extends MultiInputLayer {
    SubtractionLayer(VariablesManager variablesManager, Layer inputLayer1, Layer inputLayer2) {
        super(variablesManager, new Layer[] { inputLayer1, inputLayer2 }, inputLayer1.outputSize());
        if (inputLayer2.outputSize() != outputSize)
            throw new IllegalArgumentException("Both input layers to a subtraction layer " +
                                                       "must have the same output size.");
    }

    @Override
    Vector computeValue(NetworkState state) {
        return inputLayers[0].value(state).sub(inputLayers[1].value(state));
    }

    @Override
    Matrix localGradient(NetworkState state, Variable variable) {
        if (variable.equals(inputLayers[0].outputVariable()) || variable.equals(outputVariable))
            return Matrix.identity(outputSize);
        else if (variable.equals(inputLayers[1].outputVariable()))
            return Matrix.identity(outputSize).multiply(-1);
        else
            return Matrix.zeros(outputSize, variable.size());
    }
}
