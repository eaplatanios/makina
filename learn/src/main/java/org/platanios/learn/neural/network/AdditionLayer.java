package org.platanios.learn.neural.network;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class AdditionLayer extends MultiInputLayer {
    AdditionLayer(List<Layer> inputLayers) {
        super(inputLayers, inputLayers.get(0).outputSize());
        for (Layer layer : inputLayers)
            if (layer.outputSize() != outputSize)
                throw new IllegalArgumentException("All input layers to an addition layer " +
                                                           "must have the same output size.");
    }

    @Override
    public Vector computeValue(State state) {
        Vector value = Vectors.build(outputSize, state.vectorType());
        for (Layer layer : inputLayers)
            value.addInPlace(layer.value(state));
        return value;
    }

    @Override
    protected Matrix selfGradient(State state, Variable variable) {
        if (inputVariables.contains(variable))
            return Matrix.identity(outputSize);
        else
            return Matrix.zeros(outputSize, variable.size());
    }
}
