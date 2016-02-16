package org.platanios.learn.neural.network.activation;

import org.platanios.learn.neural.network.Layer;

/**
 * @author Emmanouil Antonios Platanios
 */
public class TanhLayer extends ActivationLayer {
    TanhLayer(Layer inputLayer) {
        super(inputLayer);
    }

    @Override
    double value(double point) {
        return Math.tanh(point);
    }

    @Override
    double gradient(double point) {
        double coshPoint = Math.cosh(point);
        double cosh2Point = Math.cosh(2 * point);
        return 4 * coshPoint * coshPoint / ((cosh2Point + 1) * (cosh2Point + 1));
    }
}
