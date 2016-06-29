package makina.math.matrix;

/**
 * @author Emmanouil Antonios Platanios
 */
public class MatrixUtilities {
    // Suppress default constructor for noninstantiability
    private MatrixUtilities() {
        throw new AssertionError();
    }

    /**
     * Computes the natural logarithm of the sum of the exponential of the values in a given vector in a way that aims
     * to avoid numerical underflow or overflow. For a vector \(\boldsymbol{x}\) with elements \(x_i\), the quantity
     * computed by this method is the following:
     * \[\log{\sum_{i}{e^{x_i}}}.\]
     *
     * @param   vector  The vector to use for this operation.
     * @return          The natural logarithm of the sum of the exponential of the values in the given vector.
     */
    public static double computeLogSumExp(Vector vector) {
        double maximumValue = vector.max();
        return maximumValue + Math.log(vector.sub(maximumValue).map(Math::exp).sum());
    }

    /**
     * Computes the natural logarithm of the sum of the exponential of the values in a given array in a way that aims to
     * avoid numerical underflow or overflow. For an array \(\boldsymbol{x}\) with elements \(x_i\), the quantity
     * computed by this method is the following:
     * \[\log{\sum_{i}{e^{x_i}}}.\]
     *
     * @param   array   The array to use for this operation.
     * @return          The natural logarithm of the sum of the exponential of the values in the given array.
     */
    public static double computeLogSumExp(double... array) {
        Vector vector = Vectors.dense(array);
        return computeLogSumExp(vector);
    }
}
