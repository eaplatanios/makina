package org.platanios.learn.math.matrix;

/**
 * @author Emmanouil Antonios Platanios
 */
public class NonPositiveDefiniteMatrixException extends Exception {
    private static final long serialVersionUID = -580851088236168017L;

    public NonPositiveDefiniteMatrixException() {
        super("The matrix is not positive definite.");
    }

    public NonPositiveDefiniteMatrixException(String message) {
        super(message);
    }
}
