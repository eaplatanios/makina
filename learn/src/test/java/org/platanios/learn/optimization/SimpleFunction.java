package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.optimization.function.AbstractFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SimpleFunction extends AbstractFunction {
    @Override
    public double computeValue(Vector point) {
        double x1 = point.get(0);
        double x2 = point.get(1);
        return 2 * Math.pow(x2 - 1, 2) + Math.pow(x1 - 2, 2) - 5;
    }

    @Override
    public Vector computeGradient(Vector point) {
        double x1 = point.get(0);
        double x2 = point.get(1);
        double dx1 = 2 * (x1 - 2);
        double dx2 = 4 * (x2 - 1);
        return Vectors.dense(dx1, dx2);
    }

    @Override
    public Matrix computeHessian(Vector point) {
        return new Matrix(new double[][] { { 2, 0 }, { 0, 4 } });
    }
}
