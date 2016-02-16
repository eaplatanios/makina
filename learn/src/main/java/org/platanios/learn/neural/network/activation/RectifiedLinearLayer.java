package org.platanios.learn.neural.network.activation;

import org.platanios.learn.neural.network.Layer;

/**
 * @author Emmanouil Antonios Platanios
 */
public class RectifiedLinearLayer extends ActivationLayer {
    private final double threshold;

    RectifiedLinearLayer(Layer inputLayer) {
        this(inputLayer, 0.0);
    }

    RectifiedLinearLayer(Layer inputLayer, double threshold) {
        super(inputLayer);
        this.threshold = threshold;
    }

    @Override
    double value(double point) {
        if (point >= threshold)
            return point;
        else
            return 0.0;
    }

    @Override
    double gradient(double point) {
        if (point >= threshold)
            return 1.0;
        else
            return 0.0;
    }
}
