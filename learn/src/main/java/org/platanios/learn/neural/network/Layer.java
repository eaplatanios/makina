package org.platanios.learn.neural.network;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.utilities.ArrayUtilities;

/**
 * @author Emmanouil Antonios Platanios
 */
abstract class Layer {
    final int outputSize;

    private Layer[] outputLayers = new Layer[0];
    private int numberOfOutputLayers = 0;

    private Matrix forwardGradient;
    private int numberOfForwardGradientsReceived = 0;

    Layer(VariablesManager variablesManager, int outputSize) {
        this.outputSize = outputSize;
    }

    abstract Layer[] inputLayers();
    abstract Variable[] inputVariables();
    abstract Variable outputVariable();

    Layer[] outputLayers() {
        return outputLayers;
    }

    int addOutputLayer(Layer layer) {
        outputLayers = ArrayUtilities.append(outputLayers, layer);
        return ++numberOfOutputLayers;
    }

    int outputSize() {
        return outputSize;
    }

    void resetForwardGradient() {
        forwardGradient = null;
        numberOfForwardGradientsReceived = 0;
    }

    Variable[] parameters() {
        return new Variable[0];
    }

    Vector value(NetworkState state) {
        Vector value = computeValue(state);
        state.set(outputVariable(), value);
        return value;
    }

    abstract Vector computeValue(NetworkState state);

    abstract Matrix localGradient(NetworkState state, Variable variable);

    Matrix recursiveGradient(NetworkState state, Variable variable) {
        Matrix gradient = localGradient(state, variable);
        for (Layer layer : inputLayers())
            if (!variable.equals(layer.outputVariable()))
                gradient.addEquals(localGradient(state, layer.outputVariable())
                                           .multiply(layer.recursiveGradient(state, variable)));
        return gradient;
    }

    Matrix[] recursiveGradient(NetworkState state, Variable... variables) {
        Matrix[] gradient = new Matrix[variables.length];
        for (int variableIndex = 0; variableIndex < variables.length; variableIndex++)
            gradient[variableIndex] = recursiveGradient(state, variables[variableIndex]);
        return gradient;
    }

    Matrix gradient(NetworkState state, Variable variable) {
        if (numberOfForwardGradientsReceived == numberOfOutputLayers || numberOfOutputLayers == 0)
            return forwardGradient.multiply(localGradient(state, variable));
        return null;
    }

    Matrix[] gradient(NetworkState state, Variable... variables) {
        if (numberOfForwardGradientsReceived == numberOfOutputLayers || numberOfOutputLayers == 0) {
            Matrix[] gradient = new Matrix[variables.length];
            for (int variableIndex = 0; variableIndex < variables.length; variableIndex++)
                gradient[variableIndex] = forwardGradient.multiply(localGradient(state, variables[variableIndex]));
            return gradient;
        }
        return null;
    }

    void backPropagateGradient(NetworkState state, Matrix forwardGradient) {
        if (numberOfForwardGradientsReceived == 0 || numberOfOutputLayers == 0)
            this.forwardGradient = forwardGradient;
        else if (numberOfForwardGradientsReceived < numberOfOutputLayers)
            this.forwardGradient.addEquals(forwardGradient);
        numberOfForwardGradientsReceived++;
        if (numberOfForwardGradientsReceived == numberOfOutputLayers || numberOfOutputLayers == 0)
            for (Layer layer : inputLayers())
                layer.backPropagateGradient(state, this.forwardGradient.multiply(localGradient(state, layer.outputVariable())));
    }
}
