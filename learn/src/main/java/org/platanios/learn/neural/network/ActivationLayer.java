package org.platanios.learn.neural.network;

import org.platanios.math.matrix.Matrix;
import org.platanios.math.matrix.Vector;
import org.platanios.math.matrix.Vectors;

/**
 * TODO: Maybe allow parameters of the different activation functions to be returned from the parameters() method of
 * this class.
 *
 * @author Emmanouil Antonios Platanios
 */
abstract class ActivationLayer extends SingleInputLayer {
    ActivationLayer(VariablesManager variablesManager, Layer inputLayer) {
        super(variablesManager, inputLayer, inputLayer.outputSize());
    }

    @Override
    int outputSize() {
        return outputSize;
    }

    @Override
    Vector computeValue(NetworkState state) {
        Vector inputValue = inputLayer.value(state);
        Vector outputValue = Vectors.build(outputSize(), state.vectorType());
        for (Vector.VectorElement element : inputValue)
            outputValue.set(element.index(), value(element.value()));
        return outputValue;
    }

    @Override
    Matrix localGradient(NetworkState state, Variable variable) {
        if (variable.equals(outputVariable)) {
            return Matrix.identity(outputSize);
        } else if (variable.equals(inputVariable)) {
            Matrix gradient = new Matrix(outputSize(), outputSize());
            Vector inputValue = inputLayer.value(state);
            for (Vector.VectorElement element : inputValue)
                gradient.setElement(element.index(), element.index(), gradient(element.value()));
            return gradient;
        } else {
            return Matrix.zeros(outputSize(), variable.size());
        }
    }

    abstract double value(double point);
    abstract double gradient(double point);
}
