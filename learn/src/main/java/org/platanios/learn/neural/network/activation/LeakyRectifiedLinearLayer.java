package org.platanios.learn.neural.network.activation;

import org.platanios.learn.neural.network.Layer;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LeakyRectifiedLinearLayer extends ActivationLayer {
    private final double threshold;
    private final double alpha;

    LeakyRectifiedLinearLayer(Layer inputLayer) {
        this(inputLayer, 0.0, 0.01);
    }

    LeakyRectifiedLinearLayer(Layer inputLayer, double alpha) {
        this(inputLayer, 0.0, alpha);
    }

    LeakyRectifiedLinearLayer(Layer inputLayer, double threshold, double alpha) {
        super(inputLayer);
        this.threshold = threshold;
        this.alpha = alpha;
    }

    @Override
    double value(double point) {
        if (point >= threshold)
            return point;
        else
            return alpha * point;
    }

    @Override
    double gradient(double point) {
        if (point >= threshold)
            return 1.0;
        else
            return alpha;
    }
}
