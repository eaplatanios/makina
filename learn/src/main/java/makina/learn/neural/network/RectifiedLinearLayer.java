package makina.learn.neural.network;

/**
 * @author Emmanouil Antonios Platanios
 */
class RectifiedLinearLayer extends ActivationLayer {
    private final double threshold;

    RectifiedLinearLayer(VariablesManager variablesManager, Layer inputLayer) {
        this(variablesManager, inputLayer, 0.0);
    }

    RectifiedLinearLayer(VariablesManager variablesManager, Layer inputLayer, double threshold) {
        super(variablesManager, inputLayer);
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
