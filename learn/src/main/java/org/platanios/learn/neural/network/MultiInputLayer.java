package org.platanios.learn.neural.network;

import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class MultiInputLayer extends Layer {
    protected final List<Layer> inputLayers;
    protected final List<Variable> inputVariables;

    protected MultiInputLayer(List<Layer> inputLayers, int outputSize) {
        super(inputLayers.stream().mapToInt(Layer::outputSize).sum(), outputSize);
        if (inputLayers.size() == 0)
            throw new IllegalArgumentException("There must exist at least one input layer for each network layer.");
        this.inputLayers = inputLayers;
        this.inputVariables =
                inputLayers.stream()
                        .map(Layer::outputVariable)
                        .collect(Collectors.toList());
    }

    @Override
    public List<Variable> inputVariables() {
        return inputVariables;
    }

    @Override
    public List<Layer> inputLayers() {
        return inputLayers;
    }
}
