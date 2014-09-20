package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorFactory;
import org.platanios.learn.optimization.function.AbstractFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
class RosenbrockFunction extends AbstractFunction {
    @Override
    public double computeValue(Vector optimizationVariables) {
        double x1 = optimizationVariables.get(0);
        double x2 = optimizationVariables.get(1);
        return 100 * Math.pow(x2 - Math.pow(x1, 2), 2) + Math.pow(1 - x1, 2);
    }

    @Override
    public Vector computeGradient(Vector optimizationVariables) {
        double x1 = optimizationVariables.get(0);
        double x2 = optimizationVariables.get(1);
        double dx1 = - 400 * (x2 - Math.pow(x1, 2)) * x1 - 2 * (1 - x1);
        double dx2 = 200 * (x2 - Math.pow(x1, 2));
        return VectorFactory.buildDense(new double[] { dx1, dx2 });
    }

    @Override
    public Matrix computeHessian(Vector optimizationVariables) {
        double x1 = optimizationVariables.get(0);
        double x2 = optimizationVariables.get(1);
        double dx1x1 = 1200 * Math.pow(x1, 2) - 400 * x2 + 2;
        double dx1x2 = - 400 * x1;
        double dx2x2 = 200;
        return new Matrix(new double[][] { { dx1x1, dx1x2 }, { dx1x2, dx2x2 } });
    }
}
