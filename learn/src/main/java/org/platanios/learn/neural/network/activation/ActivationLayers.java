package org.platanios.learn.neural.network.activation;

import org.platanios.learn.neural.network.Layer;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ActivationLayers {
    public static SigmoidLayer sigmoid(Layer inputLayer) {
        return new SigmoidLayer(inputLayer);
    }

    public static TanhLayer tanh(Layer inputLayer) {
        return new TanhLayer(inputLayer);
    }

    public static RectifiedLinearLayer rectifiedLinear(Layer inputLayer) {
        return new RectifiedLinearLayer(inputLayer);
    }

    public static RectifiedLinearLayer rectifiedLinear(Layer inputLayer, double threshold) {
        return new RectifiedLinearLayer(inputLayer, threshold);
    }

    public static LeakyRectifiedLinearLayer leakyRectifiedLinear(Layer inputLayer) {
        return new LeakyRectifiedLinearLayer(inputLayer);
    }

    public static LeakyRectifiedLinearLayer leakyRectifiedLinear(Layer inputLayer, double alpha) {
        return new LeakyRectifiedLinearLayer(inputLayer, alpha);
    }

    public static LeakyRectifiedLinearLayer leakyRectifiedLinear(Layer inputLayer, double threshold, double alpha) {
        return new LeakyRectifiedLinearLayer(inputLayer, threshold, alpha);
    }
}
