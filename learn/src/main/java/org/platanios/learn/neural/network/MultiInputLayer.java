package org.platanios.learn.neural.network;

/**
 * @author Emmanouil Antonios Platanios
 */
abstract class MultiInputLayer extends Layer {
    final Layer[] inputLayers;
    final Variable[] inputVariables;
    final Variable outputVariable;

    MultiInputLayer(VariablesManager variablesManager, Layer[] inputLayers, int outputSize) {
        super(variablesManager, outputSize);
        if (inputLayers.length == 0)
            throw new IllegalArgumentException("There must exist at least one input layer for each network layer.");
        this.inputLayers = inputLayers;
        inputVariables = new Variable[inputLayers.length];
        for (int layerIndex = 0; layerIndex < inputLayers.length; layerIndex++) {
            inputVariables[layerIndex] = inputLayers[layerIndex].outputVariable();
            inputLayers[layerIndex].addOutputLayer(this);
        }
        outputVariable = variablesManager.layerVariable(this);
    }

    @Override
    Layer[] inputLayers() {
        return inputLayers;
    }

    @Override
    Variable[] inputVariables() {
        return inputVariables;
    }

    @Override
    Variable outputVariable() {
        return outputVariable;
    }
}
