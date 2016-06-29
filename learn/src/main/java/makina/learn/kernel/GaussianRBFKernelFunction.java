package makina.learn.kernel;

import makina.math.matrix.Vector;
import makina.math.matrix.VectorNorm;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GaussianRBFKernelFunction implements KernelFunction<Vector> {
    private final double gamma;

    public GaussianRBFKernelFunction(double gamma) {
        this.gamma = gamma;
    }

    @Override
    public double getValue(Vector instance1, Vector instance2) {
        Vector difference = instance1.sub(instance2);
        return Math.exp(-gamma * difference.norm(VectorNorm.L2_SQUARED));
    }
}
