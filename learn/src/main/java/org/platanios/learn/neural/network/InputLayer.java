package org.platanios.learn.neural.network;

import com.google.common.collect.Lists;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class InputLayer extends Layer {
    private final VectorVariable inputVariable;

    InputLayer(int size) {
        super(size, size);
        inputVariable = Variables.vectorVariable(size);
    }

    public Variable inputVariable() {
        return inputVariable;
    }

    @Override
    public List<Variable> inputVariables() {
        return Lists.newArrayList(inputVariable);
    }

    @Override
    public List<Layer> inputLayers() {
        return new ArrayList<>();
    }

    @Override
    public Vector computeValue(State state) {
        return state.get(inputVariable);
    }

    @Override
    public Matrix gradient(State state, Variable variable) {
        if (inputVariable.equals(variable) || outputVariable.equals(variable))
            return Matrix.identity(inputVariable.size());
        else
            return Matrix.zeros(inputSize(), variable.size());
    }

    @Override
    protected Matrix selfGradient(State state, Variable variable) {
        return null;
    }
}
