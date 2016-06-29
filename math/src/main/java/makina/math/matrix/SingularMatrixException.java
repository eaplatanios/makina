package makina.math.matrix;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SingularMatrixException extends Exception {
    private static final long serialVersionUID = 2661298174206787506L;

    public SingularMatrixException() {
        super("The matrix is singular.");
    }

    public SingularMatrixException(String message) {
        super(message);
    }
}
