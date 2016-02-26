package org.platanios.learn.neural.activation;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SigmoidFunction {
    public static Vector value(Vector point) {
        Vector value = Vectors.build(point.size(), point.type());
        for (Vector.VectorElement element : point)
            value.set(element.index(), value(element.value()));
        return value;
    }

    public static Matrix gradient(Vector point) {
        Matrix gradient = new Matrix(point.size(), point.size());
        for (Vector.VectorElement element : point)
            gradient.setElement(element.index(), element.index(), gradient(element.value()));
        return gradient;
    }

    public static double value(double point) {
        return 1 / (1 + Math.exp(-point));
    }

    public static double gradient(double point) {
        double value = value(point);
        return value * (1 - value);
    }
}
