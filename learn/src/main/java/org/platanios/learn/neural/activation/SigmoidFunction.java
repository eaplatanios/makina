package org.platanios.learn.neural.activation;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SigmoidFunction implements ActivationFunction {
    public SigmoidFunction() { }

    @Override
    public Vector getValue(Vector point) {
        Vector value = Vectors.build(point.size(), point.type());
        for (Vector.VectorElement element : point)
            value.set(element.index(), getValue(element.value()));
        return value;
    }

    @Override
    public Matrix getGradient(Vector point) {
        Matrix gradient = new Matrix(point.size(), point.size());
        for (Vector.VectorElement element : point)
            gradient.setElement(element.index(), element.index(), getGradient(element.value()));
        return gradient;
    }

    private double getValue(double point) {
        return 1 / (1 + Math.exp(-point));
    }

    private double getGradient(double point) {
        double value = getValue(point);
        return value * (1 - value);
    }
}
