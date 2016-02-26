package org.platanios.learn.neural.network;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.utilities.ArrayUtilities;

import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class NetworkBuilder {
    final VariablesManager variablesManager = new VariablesManager();
    final LayersManager layersManager = new LayersManager();
    final List<Layer> layers = new ArrayList<>();
    final List<Layer> inputLayers = new ArrayList<>();
    final Set<Variable> variables = new HashSet<>();
    final Set<Variable> parameters = new HashSet<>();

    Layer outputLayer;

    public NetworkBuilder() { }

    public int addConstantLayer(Vector value) {
        return addLayer(new ConstantLayer(variablesManager, value));
    }

    public int addInputLayer(int size) {
        InputLayer layer = new InputLayer(variablesManager, size);
        inputLayers.add(layer);
        return addLayer(layer);
    }

    public int addInputLayer(int size, String name) {
        InputLayer layer = new InputLayer(variablesManager, name, size);
        inputLayers.add(layer);
        return addLayer(layer);
    }

    public int addFullyConnectedLayer(int inputLayerId, int numberOfHiddenUnits) {
        return addLayer(new FullyConnectedLayer(variablesManager, layersManager.get(inputLayerId), numberOfHiddenUnits));
    }

    public int addFullyConnectedLayer(boolean output, int inputLayerId, int numberOfHiddenUnits) {
        return addLayer(new FullyConnectedLayer(variablesManager, layersManager.get(inputLayerId), numberOfHiddenUnits), output);
    }

    public int addFullyConnectedLayer(int inputLayerId, int numberOfHiddenUnits, boolean includeBiasTerm) {
        return addLayer(new FullyConnectedLayer(variablesManager,
                                                layersManager.get(inputLayerId),
                                                numberOfHiddenUnits,
                                                includeBiasTerm));
    }

    public int addFullyConnectedLayer(boolean output,
                                      int inputLayerId,
                                      int numberOfHiddenUnits,
                                      boolean includeBiasTerm) {
        return addLayer(new FullyConnectedLayer(variablesManager,
                                                layersManager.get(inputLayerId),
                                                numberOfHiddenUnits,
                                                includeBiasTerm), output);
    }

    public int addFullyConnectedLayer(int inputLayerId,
                                      int numberOfHiddenUnits,
                                      String weightsVariableName) {
        return addLayer(new FullyConnectedLayer(variablesManager,
                                                layersManager.get(inputLayerId),
                                                numberOfHiddenUnits,
                                                weightsVariableName));
    }

    public int addFullyConnectedLayer(boolean output,
                                      int inputLayerId,
                                      int numberOfHiddenUnits,
                                      String weightsVariableName) {
        return addLayer(new FullyConnectedLayer(variablesManager,
                                                layersManager.get(inputLayerId),
                                                numberOfHiddenUnits,
                                                weightsVariableName), output);
    }

    public int addFullyConnectedLayer(int inputLayerId,
                                      int numberOfHiddenUnits,
                                      String weightsVariableName,
                                      String biasVariableName) {
        return addLayer(new FullyConnectedLayer(variablesManager,
                                                layersManager.get(inputLayerId),
                                                numberOfHiddenUnits,
                                                weightsVariableName,
                                                biasVariableName));
    }

    public int addFullyConnectedLayer(boolean output,
                                      int inputLayerId,
                                      int numberOfHiddenUnits,
                                      String weightsVariableName,
                                      String biasVariableName) {
        return addLayer(new FullyConnectedLayer(variablesManager,
                                                layersManager.get(inputLayerId),
                                                numberOfHiddenUnits,
                                                weightsVariableName,
                                                biasVariableName), output);
    }

    public int addSigmoidLayer(int inputLayerId) {
        return addLayer(new SigmoidLayer(variablesManager, layersManager.get(inputLayerId)));
    }

    public int addSigmoidLayer(boolean output, int inputLayerId) {
        return addLayer(new SigmoidLayer(variablesManager, layersManager.get(inputLayerId)), output);
    }

    public int addTanhLayer(int inputLayerId) {
        return addLayer(new TanhLayer(variablesManager, layersManager.get(inputLayerId)));
    }

    public int addTanhLayer(boolean output, int inputLayerId) {
        return addLayer(new TanhLayer(variablesManager, layersManager.get(inputLayerId)), output);
    }

    public int addRectifiedLinearLayer(int inputLayerId) {
        return addLayer(new RectifiedLinearLayer(variablesManager, layersManager.get(inputLayerId)));
    }

    public int addRectifiedLinearLayer(boolean output, int inputLayerId) {
        return addLayer(new RectifiedLinearLayer(variablesManager, layersManager.get(inputLayerId)), output);
    }

    public int addRectifiedLinearLayer(int inputLayerId, double threshold) {
        return addLayer(new RectifiedLinearLayer(variablesManager, layersManager.get(inputLayerId), threshold));
    }

    public int addRectifiedLinearLayer(boolean output, int inputLayerId, double threshold) {
        return addLayer(new RectifiedLinearLayer(variablesManager, layersManager.get(inputLayerId), threshold), output);
    }

    public int addLeakyRectifiedLinearLayer(int inputLayerId) {
        return addLayer(new LeakyRectifiedLinearLayer(variablesManager, layersManager.get(inputLayerId)));
    }

    public int addLeakyRectifiedLinearLayer(boolean output, int inputLayerId) {
        return addLayer(new LeakyRectifiedLinearLayer(variablesManager, layersManager.get(inputLayerId)), output);
    }

    public int addLeakyRectifiedLinearLayer(int inputLayerId, double alpha) {
        return addLayer(new LeakyRectifiedLinearLayer(variablesManager, layersManager.get(inputLayerId), alpha));
    }

    public int addLeakyRectifiedLinearLayer(boolean output,int inputLayerId, double alpha) {
        return addLayer(new LeakyRectifiedLinearLayer(variablesManager, layersManager.get(inputLayerId), alpha), output);
    }

    public int addLeakyRectifiedLinearLayer(int inputLayerId, double threshold, double alpha) {
        return addLayer(new LeakyRectifiedLinearLayer(variablesManager, layersManager.get(inputLayerId), threshold, alpha));
    }

    public int addLeakyRectifiedLinearLayer(boolean output,
                                            int inputLayerId,
                                            double threshold,
                                            double alpha) {
        return addLayer(new LeakyRectifiedLinearLayer(variablesManager, layersManager.get(inputLayerId), threshold, alpha), output);
    }

    public int addAdditionLayer(int... inputLayerIds) {
        Layer[] inputLayers = new Layer[inputLayerIds.length];
        for (int layerIndex = 0; layerIndex < inputLayerIds.length; layerIndex++)
            inputLayers[layerIndex] = layersManager.get(inputLayerIds[layerIndex]);
        return addLayer(new AdditionLayer(variablesManager, inputLayers));
    }

    public int addAdditionLayer(boolean output, int... inputLayerIds) {
        Layer[] inputLayers = new Layer[inputLayerIds.length];
        for (int layerIndex = 0; layerIndex < inputLayerIds.length; layerIndex++)
            inputLayers[layerIndex] = layersManager.get(inputLayerIds[layerIndex]);
        return addLayer(new AdditionLayer(variablesManager, inputLayers), output);
    }

    public int addSubtractionLayer(int inputLayer1Id, int inputLayer2Id) {
        return addLayer(new SubtractionLayer(variablesManager, layersManager.get(inputLayer1Id), layersManager.get(inputLayer2Id)));
    }

    public int addSubtractionLayer(boolean output, int inputLayer1Id, int inputLayer2Id) {
        return addLayer(new SubtractionLayer(variablesManager, layersManager.get(inputLayer1Id), layersManager.get(inputLayer2Id)), output);
    }

    public int addElementwiseMultiplicationLayer(int... inputLayerIds) {
        Layer[] inputLayers = new Layer[inputLayerIds.length];
        for (int layerIndex = 0; layerIndex < inputLayerIds.length; layerIndex++)
            inputLayers[layerIndex] = layersManager.get(inputLayerIds[layerIndex]);
        return addLayer(new ElementwiseMultiplicationLayer(variablesManager, inputLayers));
    }

    public int addElementwiseMultiplicationLayer(boolean output, int... inputLayerIds) {
        Layer[] inputLayers = new Layer[inputLayerIds.length];
        for (int layerIndex = 0; layerIndex < inputLayerIds.length; layerIndex++)
            inputLayers[layerIndex] = layersManager.get(inputLayerIds[layerIndex]);
        return addLayer(new ElementwiseMultiplicationLayer(variablesManager, inputLayers), output);
    }

    private int addLayer(Layer layer) {
        return addLayer(layer, false);
    }

    private int addLayer(Layer layer, boolean output) {
        if (layers.contains(layer))
            throw new IllegalArgumentException("The provided layer has already been added to this network builder.");
        if (outputLayer != null && output)
            throw new IllegalArgumentException("There can only be one output layer for each network.");
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++)  // Make sure the added layers are sorted such that layers that are used
            if (ArrayUtilities.contains(layers.get(layerIndex).inputLayers(), layer)) {     // as inputs for other layers, come before them.
                layers.add(layerIndex, layer);
                break;
            }
        layers.add(layer);
        variables.addAll(Arrays.asList(layer.inputVariables()));
        variables.add(layer.outputVariable());
        if (output)
            outputLayer = layer;
        return layersManager.addLayer(layer);
    }

    public Network build() {
        if (outputLayer == null)
            throw new IllegalStateException("A network cannot be built without an output layer.");
        return new Network(this);
    }
}
