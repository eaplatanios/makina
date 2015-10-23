package org.platanios.learn.kernel;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LinearKernelFunction implements KernelFunction<Vector> {
    private final double shift;

    public LinearKernelFunction() {
        this.shift = 0;
    }

    public LinearKernelFunction(double shift) {
        this.shift = shift;
    }

    @Override
    public double getValue(Vector instance1, Vector instance2) {
        return instance1.dot(instance2) + shift;
    }
}
