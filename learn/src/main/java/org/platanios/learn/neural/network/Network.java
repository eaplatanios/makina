package org.platanios.learn.neural.network;

import com.google.common.collect.Lists;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Network extends MultiInputLayer {
    private final Set<Variable> variables;
    private final Set<Variable> parameters;
    private final Layer outputLayer;

    public static class Builder {
        private final Set<Layer> layers = new HashSet<>();
        private final Set<Variable> variables = new HashSet<>();
        private final Set<Variable> parameters = new HashSet<>();

        private final List<Layer> inputLayers;

        private OutputLayer outputLayer;

        public Builder(Layer inputLayer) {
            this(Lists.newArrayList(inputLayer));
        }

        public Builder(List<Layer> inputLayers) {
            layers.addAll(inputLayers);
            variables.addAll(inputLayers.stream()
                                     .map(Layer::inputVariables)
                                     .flatMap(List::stream)
                                     .collect(Collectors.toList()));
            variables.addAll(inputLayers.stream()
                                     .map(Layer::outputVariable)
                                     .collect(Collectors.toList()));
            this.inputLayers = inputLayers;
        }

        public Builder addLayer(Layer layer) {
            if (!layers.containsAll(layer.inputLayers()))
                throw new IllegalStateException("The input layers of the layer being added must " +
                                                        "already be part of the network.");
            if (outputLayer != null)
                throw new IllegalStateException("No layers can be added to the network after an " +
                                                        "output layer has been added.");
            layers.add(layer);
            variables.add(layer.outputVariable);
            parameters.addAll(layer.parameters());
            if (layer instanceof OutputLayer)
                outputLayer = (OutputLayer) layer;
            return this;
        }

        public Builder addLayer(Layer layer, boolean output) {
            addLayer(layer);
            if (output && !(layer instanceof OutputLayer))
                addLayer(new OutputLayer(layer));
            else if (output)
                outputLayer = (OutputLayer) layer;
            return this;
        }

        public Builder addOutputLayer(SingleInputLayer layer) {
            return addLayer(layer, true);
        }

        public Network build() {
            if (outputLayer == null)
                throw new IllegalStateException("The network cannot be built without an output layer.");
            return new Network(this);
        }
    }

    private Network(Builder builder) {
        super(builder.inputLayers, builder.outputLayer.outputSize);
        variables = builder.variables;
        parameters = builder.parameters;
        outputLayer = builder.outputLayer;
    }

    @Override
    public Set<Variable> parameters() {
        return parameters;
    }

    public Set<Variable> variables() {
        return variables;
    }

    @Override
    public Vector computeValue(State state) {
        return outputLayer.value(state);
    }

    @Override
    public Matrix gradient(State state, Variable variable) {
        return outputLayer.gradient(state, variable);
    }

    @Override
    protected Matrix selfGradient(State state, Variable variable) {
        return null;
    }

    @Override
    public List<Matrix> gradient(State state, List<Variable> variables) {
        return outputLayer.gradient(state, variables);
    }
}
