package org.platanios.learn.neural.network;

/**
 * @author Emmanouil Antonios Platanios
 */
class LeakyRectifiedLinearLayer extends ActivationLayer {
    private final double threshold;
    private final double alpha;

    LeakyRectifiedLinearLayer(VariablesManager variablesManager, Layer inputLayer) {
        this(variablesManager, inputLayer, 0.0, 0.01);
    }

    LeakyRectifiedLinearLayer(VariablesManager variablesManager, Layer inputLayer, double alpha) {
        this(variablesManager, inputLayer, 0.0, alpha);
    }

    LeakyRectifiedLinearLayer(VariablesManager variablesManager, Layer inputLayer, double threshold, double alpha) {
        super(variablesManager, inputLayer);
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
