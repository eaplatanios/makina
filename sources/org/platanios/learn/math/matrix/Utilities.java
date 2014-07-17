package org.platanios.learn.math.matrix;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Utilities {
    // Suppress default constructor for noninstantiability
    private Utilities() {
        throw new AssertionError();
    }

    /**
     * Computes the natural logarithm of the sum of the exponential of the values in a given vector in a way that aims
     * to avoid numerical underflow or overflow.
     *
     * @param   vector  The vector to use for this operation.
     * @return          The natural logarithm of the sum of the exponential of the values in the given vector.
     */
    public static double computeLogSumExp(Vector vector) {
        double maximumValue = vector.getMaximumValue();
        return maximumValue + Math.log(vector.subtract(maximumValue).computeFunctionResult(Math::exp).computeSum());
    }
}
