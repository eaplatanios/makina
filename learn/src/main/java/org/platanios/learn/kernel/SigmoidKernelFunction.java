package org.platanios.learn.kernel;

import org.platanios.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SigmoidKernelFunction implements KernelFunction<Vector> {
    private final double scale;
    private final double shift;

    public SigmoidKernelFunction(double scale, double shift) {
        this.scale = scale;
        this.shift = shift;
    }

    @Override
    public double getValue(Vector instance1, Vector instance2) {
        return Math.tanh(scale * instance1.dot(instance2) + shift);
    }
}
