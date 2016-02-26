package org.platanios.learn.neural.network;

/**
 * @author Emmanouil Antonios Platanios
 */
class SigmoidLayer extends ActivationLayer {
    SigmoidLayer(VariablesManager variablesManager, Layer inputLayer) {
        super(variablesManager, inputLayer);
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
