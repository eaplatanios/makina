package org.platanios.learn.neural.network;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class Layer {
    protected final int inputSize;
    protected final int outputSize;
    protected final Variable outputVariable;

    protected Layer(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.outputVariable = Variables.layerVariable(this);
    }

    public int inputSize() {
        return inputSize;
    }

    public int outputSize() {
        return outputSize;
    }

    public abstract List<Variable> inputVariables();
    public abstract List<Layer> inputLayers();

    public Variable outputVariable() {
        return outputVariable;
    }

    public Set<Variable> parameters() {
        return new HashSet<>();
    }

    public Vector value(State state) {
        Vector value = computeValue(state);
        state.set(outputVariable, value);
        return value;
    }

    protected abstract Vector computeValue(State state);

    public Matrix gradient(State state, Variable variable) {
        Matrix gradient = selfGradient(state, variable);
        inputLayers().stream()
                .filter(inputLayer -> !variable.equals(inputLayer.outputVariable()))
                .forEach(inputLayer -> gradient.addEquals(selfGradient(state, inputLayer.outputVariable())
                                                                  .multiply(inputLayer.gradient(state, variable))));
        return gradient;
    }

    protected abstract Matrix selfGradient(State state, Variable variable);

    public List<Matrix> gradient(State state, List<Variable> variables) {
        return variables.stream()
                .map(variable -> gradient(state, variable))
                .collect(Collectors.toList());
    }
}
