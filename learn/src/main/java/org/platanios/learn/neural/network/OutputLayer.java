package org.platanios.learn.neural.network;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class OutputLayer extends SingleInputLayer {
    OutputLayer(Layer inputLayer) {
        super(inputLayer, inputLayer.outputSize);
    }

    @Override
    public Vector computeValue(State state) {
        return inputLayer.value(state);
    }

    @Override
    public Matrix gradient(State state, Variable variable) {
        return inputLayer.gradient(state, variable);
    }

    @Override
    protected Matrix selfGradient(State state, Variable variable) {
        return null;
    }
}
