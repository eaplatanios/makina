package org.platanios.learn.neural.network;

import com.google.common.collect.Lists;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class SingleInputLayer extends Layer {
    protected final Layer inputLayer;
    protected final Variable inputVariable;

    protected SingleInputLayer(Layer inputLayer, int outputSize) {
        super(inputLayer.outputSize(), outputSize);
        this.inputLayer = inputLayer;
        this.inputVariable = inputLayer.outputVariable();
    }

    public Variable inputVariable() {
        return inputVariable;
    }

    public Layer inputLayer() {
        return inputLayer;
    }

    @Override
    public List<Variable> inputVariables() {
        return Lists.newArrayList(inputVariable);
    }

    @Override
    public List<Layer> inputLayers() {
        return Lists.newArrayList(inputLayer);
    }
}
