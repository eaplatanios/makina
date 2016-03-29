package org.platanios.learn.kernel;

import org.platanios.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class PolynomialKernelFunction implements KernelFunction<Vector> {
    private final double shift;
    private final double power;

    public PolynomialKernelFunction(double shift, double power) {
        this.shift = shift;
        this.power = power;
    }

    @Override
    public double getValue(Vector instance1, Vector instance2) {
        return Math.pow(instance1.dot(instance2) + shift, power);
    }
}
