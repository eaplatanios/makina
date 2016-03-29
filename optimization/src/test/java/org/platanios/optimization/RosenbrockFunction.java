package org.platanios.optimization;

import org.platanios.math.matrix.Matrix;
import org.platanios.math.matrix.Vector;
import org.platanios.math.matrix.Vectors;
import org.platanios.optimization.function.AbstractFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
class RosenbrockFunction extends AbstractFunction {
    @Override
    public double computeValue(Vector point) {
        double x1 = point.get(0);
        double x2 = point.get(1);
        return 100 * Math.pow(x2 - Math.pow(x1, 2), 2) + Math.pow(1 - x1, 2);
    }

    @Override
    public Vector computeGradient(Vector point) {
        double x1 = point.get(0);
        double x2 = point.get(1);
        double dx1 = - 400 * (x2 - Math.pow(x1, 2)) * x1 - 2 * (1 - x1);
        double dx2 = 200 * (x2 - Math.pow(x1, 2));
        return Vectors.dense(dx1, dx2);
    }

    @Override
    public Matrix computeHessian(Vector point) {
        double x1 = point.get(0);
        double x2 = point.get(1);
        double dx1x1 = 1200 * Math.pow(x1, 2) - 400 * x2 + 2;
        double dx1x2 = - 400 * x1;
        double dx2x2 = 200;
        return new Matrix(new double[][] { { dx1x1, dx1x2 }, { dx1x2, dx2x2 } });
    }
}
