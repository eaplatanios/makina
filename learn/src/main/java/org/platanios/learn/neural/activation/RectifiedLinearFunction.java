package org.platanios.learn.neural.activation;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class RectifiedLinearFunction implements ActivationFunction {
    private final double threshold;

    public static class Builder {
        private double threshold = 0.0;

        public Builder() { }

        public Builder(double threshold) {
            this.threshold = threshold;
        }

        public Builder threshold(double threshold) {
            this.threshold = threshold;
            return this;
        }

        public RectifiedLinearFunction build() {
            return new RectifiedLinearFunction(this);
        }
    }

    private RectifiedLinearFunction(Builder builder) {
        threshold = builder.threshold;
    }

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
        if (point >= threshold)
            return point;
        else
            return 0.0;
    }

    private double getGradient(double point) {
        if (point >= threshold)
            return 1.0;
        else
            return 0.0;
    }
}
