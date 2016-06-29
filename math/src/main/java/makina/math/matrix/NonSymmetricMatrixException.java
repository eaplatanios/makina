package makina.math.matrix;

/**
 * @author Emmanouil Antonios Platanios
 */
public class NonSymmetricMatrixException extends Exception {
    private static final long serialVersionUID = 9153953427416283275L;

    public NonSymmetricMatrixException() {
        super("The matrix is not symmetric.");
    }

    public NonSymmetricMatrixException(String message) {
        super(message);
    }
}
