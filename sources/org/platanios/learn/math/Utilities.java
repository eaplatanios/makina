package org.platanios.learn.math;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Utilities {
    /**
     * Computes the machine epsilon in double precision.
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
     * Computes the square root of the sum of the squares of two numbers without having an underflow or an overflow.
     * Denoting the two numbers by \(a\) and \(b\), respectively, this function computes the quantity:
     * \[\sqrt{a^2+b^2}.\]
     *
     * @param   a   The first number.
     * @param   b   The second number.
     * @return      The square root of the sum of the squares of the two provided numbers.
     */
    public static double computeSquareRootOfSumOfSquares(double a, double b) {
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
