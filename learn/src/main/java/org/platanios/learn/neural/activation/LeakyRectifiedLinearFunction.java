package org.platanios.learn.neural.activation;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LeakyRectifiedLinearFunction implements ActivationFunction {
    private final double threshold;
    private final double alpha;

    public static class Builder {
        private double threshold = 0.0;
        private double alpha = 0.01;

        public Builder() { }

        public Builder(double threshold) {
            this.threshold = threshold;
        }

        public Builder(double threshold, double alpha) {
            this.alpha = alpha;
        }

        public Builder threshold(double threshold) {
            this.threshold = threshold;
            return this;
        }

        public Builder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }

        public LeakyRectifiedLinearFunction build() {
            return new LeakyRectifiedLinearFunction(this);
        }
    }

    private LeakyRectifiedLinearFunction(Builder builder) {
        threshold = builder.threshold;
        alpha = builder.alpha;
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
            return alpha * point;
    }

    private double getGradient(double point) {
        if (point >= threshold)
            return 1.0;
        else
            return alpha;
    }
}
