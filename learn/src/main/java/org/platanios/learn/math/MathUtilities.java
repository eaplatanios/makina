package org.platanios.learn.math;

/**
 * @author Emmanouil Antonios Platanios
 */
public class MathUtilities {
    // Suppress default constructor for noninstantiability
    private MathUtilities() {
        throw new AssertionError();
    }

    /**
     * Computes the argmax of a given array. This method returns the index of the element with the maximum computeValue within
     * the given array.
     *
     * @param   array   The array to use.
     * @return          The index of the element with the maximum computeValue within the given array.
     */
    public static int computeArgMax(double[] array) {
        double maximumValue = Double.MIN_VALUE;
        int maximumValueIndex = -1;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > maximumValue) {
                maximumValue = array[i];
                maximumValueIndex = i;
            }
        }
        return maximumValueIndex;
    }

    /**
     * Computes the machine epsilon in double precision.
     * TODO: We can use the Java Double.MIN_VALUE instead of using this calculation.
     *
     * @return  The machine epsilon in double precision.
     */
    public static double computeMachineEpsilonDouble() {
        double epsilon = 1;
        while (1 + epsilon / 2 > 1.0) {
            epsilon /= 2;
        }
        return epsilon;
    }

    /**
     * Computes the square root of the sum of the squares of two numbers (that is equivalent to computing length of the
     * hypotenuse of a right triangle given the lengths of the other two sides) without having an underflow or an
     * overflow. Denoting the two numbers by \(a\) and \(b\), respectively, this function computes the quantity:
     * \[\sqrt{a^2+b^2}.\]
     *
     * @param   a   The first number.
     * @param   b   The second number.
     * @return      The square root of the sum of the squares of the two provided numbers.
     */
    public static double computeHypotenuse(double a, double b) {
        double result;
        if (Math.abs(a) > Math.abs(b)) {
            result = b / a;
            result = Math.abs(a) * Math.sqrt(1 + result * result);
        } else if (b != 0) {
            result = a / b;
            result = Math.abs(b) * Math.sqrt(1 + result * result);
        } else {
            result = 0.0;
        }
        return result;
    }
}
