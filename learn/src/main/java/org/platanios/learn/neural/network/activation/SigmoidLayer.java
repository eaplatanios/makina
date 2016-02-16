package org.platanios.learn.neural.network.activation;

import org.platanios.learn.neural.network.Layer;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SigmoidLayer extends ActivationLayer {
    SigmoidLayer(Layer inputLayer) {
        super(inputLayer);
    }

    @Override
    double value(double point) {
        return 1 / (1 + Math.exp(-point));
    }

    @Override
    double gradient(double point) {
        double value = value(point);
        return value * (1 - value);
    }
}
