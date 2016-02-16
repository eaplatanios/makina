package org.platanios.learn.neural.network.activation;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.neural.network.Layer;
import org.platanios.learn.neural.network.SingleInputLayer;
import org.platanios.learn.neural.network.State;
import org.platanios.learn.neural.network.Variable;

/**
 * TODO: Maybe allow parameters of the different activation functions to be returned from the parameters() method of
 * this class.
 *
 * @author Emmanouil Antonios Platanios
 */
public abstract class ActivationLayer extends SingleInputLayer {
    ActivationLayer(Layer inputLayer) {
        super(inputLayer, inputLayer.outputSize());
    }

    @Override
    public int outputSize() {
        return inputSize;
    }

    @Override
    public Vector computeValue(State state) {
        Vector inputValue = inputLayer.value(state);
        Vector outputValue = Vectors.build(outputSize(), state.vectorType());
        for (Vector.VectorElement element : inputValue)
            outputValue.set(element.index(), value(element.value()));
        return outputValue;
    }

    @Override
    protected Matrix selfGradient(State state, Variable variable) {
        if (variable.equals(inputVariable)) {
            Matrix gradient = new Matrix(outputSize(), inputSize);
            Vector inputValue = inputLayer.value(state);
            for (Vector.VectorElement element : inputValue)
                gradient.setElement(element.index(), element.index(), gradient(element.value()));
            return gradient;
        } else {
            return Matrix.zeros(inputSize, variable.size());
        }
    }

    abstract double value(double point);
    abstract double gradient(double point);
}
