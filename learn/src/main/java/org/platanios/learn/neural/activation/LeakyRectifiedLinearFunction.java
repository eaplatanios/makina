package org.platanios.learn.neural.activation;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LeakyRectifiedLinearFunction {
    private final double threshold;
    private final double alpha;

    public LeakyRectifiedLinearFunction() {
        this(0.0, 0.01);
    }

    public LeakyRectifiedLinearFunction(double alpha) {
        this(0.0, alpha);
    }

    public LeakyRectifiedLinearFunction(double threshold, double alpha) {
        this.threshold = threshold;
        this.alpha = alpha;
    }

    public Vector value(Vector point) {
        Vector value = Vectors.build(point.size(), point.type());
        for (Vector.VectorElement element : point)
            value.set(element.index(), value(element.value()));
        return value;
    }

    public Matrix gradient(Vector point) {
        Matrix gradient = new Matrix(point.size(), point.size());
        for (Vector.VectorElement element : point)
            gradient.setElement(element.index(), element.index(), gradient(element.value()));
        return gradient;
    }

    public double value(double point) {
        if (point >= threshold)
            return point;
        else
            return alpha * point;
    }

    public double gradient(double point) {
        if (point >= threshold)
            return 1.0;
        else
            return alpha;
    }
}
