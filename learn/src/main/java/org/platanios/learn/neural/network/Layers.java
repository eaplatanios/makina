package org.platanios.learn.neural.network;

import org.platanios.learn.neural.network.activation.*;

import java.util.Arrays;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Layers {
    public static InputLayer input(int size) {
        return new InputLayer(size);
    }

    public static OutputLayer output(Layer inputLayer) {
        return new OutputLayer(inputLayer);
    }

    public static FullyConnectedLayer fullyConnected(Layer inputLayer, int numberOfHiddenUnits) {
        return new FullyConnectedLayer(inputLayer, numberOfHiddenUnits);
    }

    public static FullyConnectedLayer fullyConnected(Layer inputLayer, int numberOfHiddenUnits, boolean includeBiasTerm) {
        return new FullyConnectedLayer(inputLayer, numberOfHiddenUnits, includeBiasTerm);
    }

    public static FullyConnectedLayer fullyConnected(Layer inputLayer, int numberOfHiddenUnits, String weightsVariableName) {
        return new FullyConnectedLayer(inputLayer, numberOfHiddenUnits, weightsVariableName);
    }

    public static FullyConnectedLayer fullyConnected(Layer inputLayer, int numberOfHiddenUnits, String weightsVariableName, String biasVariableName) {
        return new FullyConnectedLayer(inputLayer, numberOfHiddenUnits, weightsVariableName, biasVariableName);
    }

    public static SigmoidLayer sigmoidActivation(Layer inputLayer) {
        return ActivationLayers.sigmoid(inputLayer);
    }

    public static TanhLayer tanhActivation(Layer inputLayer) {
        return ActivationLayers.tanh(inputLayer);
    }

    public static RectifiedLinearLayer RectifiedLinearActivation(Layer inputLayer) {
        return ActivationLayers.rectifiedLinear(inputLayer);
    }

    public static RectifiedLinearLayer RectifiedLinearActivation(Layer inputLayer, double threshold) {
        return ActivationLayers.rectifiedLinear(inputLayer, threshold);
    }

    public static LeakyRectifiedLinearLayer leakyRectifiedActivation(Layer inputLayer) {
        return ActivationLayers.leakyRectifiedLinear(inputLayer);
    }

    public static LeakyRectifiedLinearLayer leakyRectifiedActivation(Layer inputLayer, double alpha) {
        return ActivationLayers.leakyRectifiedLinear(inputLayer, alpha);
    }

    public static LeakyRectifiedLinearLayer leakyRectifiedActivation(Layer inputLayer, double threshold, double alpha) {
        return ActivationLayers.leakyRectifiedLinear(inputLayer, threshold, alpha);
    }

    public static AdditionLayer addition(List<Layer> inputLayers) {
        return new AdditionLayer(inputLayers);
    }

    public static AdditionLayer addition(Layer... inputLayers) {
        return new AdditionLayer(Arrays.asList(inputLayers));
    }
}
