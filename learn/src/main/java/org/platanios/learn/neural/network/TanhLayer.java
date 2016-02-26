package org.platanios.learn.neural.network;

/**
 * @author Emmanouil Antonios Platanios
 */
class TanhLayer extends ActivationLayer {
    TanhLayer(VariablesManager variablesManager, Layer inputLayer) {
        super(variablesManager, inputLayer);
    }

    @Override
    double value(double point) {
        return Math.tanh(point);
    }

    @Override
    double gradient(double point) {
        double tanhPoint = Math.tanh(point);
        return 1 - tanhPoint * tanhPoint;
    }
}
