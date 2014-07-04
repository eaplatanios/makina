package org.platanios.learn.math.matrix;

/**
 * Implements the Cholesky decomposition algorithm for matrix \(A\). This algorithm attempts to find a lower diagonal
 * matrix \(L\), such that \(A=LL^T\). The algorithm determines whether or not \(A\) is symmetric and positive definite
 * and the decomposition is actually only possible is \(A\) is symmetric and positive definite. If it is not, the result
 * is meaningless and should not be used in further computations.
 *
 * @author Emmanouil Antonios Platanios
 */
public class CholeskyDecomposition {
    /** The lower triangular Cholesky factor represented as a two-dimensional array. */
    private final double[][] L;
    /** The dimension of the matrix whose decomposition is being computed (that is, \(A\)). The Cholesky decomposition
     * requires a square matrix and so the row dimension and the column dimension of \(A\) are equal. */
    private final int dimension;

    /** A boolean value indicating whether the matrix whose decomposition is computed is symmetric and positive definite
     * or not. */
    private boolean isSymmetricAndPositiveDefinite;

    /**
     * Constructs a Cholesky decomposition object for the provided matrix. The actual decomposition is computed within
     * this constructor.
     *
     * @param   matrix  The matrix whose Cholesky decomposition is being computed.
     */
    public CholeskyDecomposition(Matrix matrix) {
        dimension = matrix.getRowDimension();
        isSymmetricAndPositiveDefinite = matrix.getColumnDimension() == dimension;
        if (!isSymmetricAndPositiveDefinite) {
            throw new IllegalArgumentException("The matrix has to be square.");
        }
        double[][] matrixArray = matrix.getArray();
        L = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            double d = 0.0;
            for (int j = 0; j < i; j++) {
                double s = 0.0;
                for (int k = 0; k < j; k++) {
                    s += L[j][k] * L[i][k];
                }
                L[i][j] = (matrixArray[i][j] - s)/L[j][j];
                d = d + L[i][j] * L[i][j];
                isSymmetricAndPositiveDefinite &= (matrixArray[j][i] == matrixArray[i][j]);
            }
            d -= matrixArray[i][i];
            isSymmetricAndPositiveDefinite &= (d < 0.0);
            L[i][i] = Math.sqrt(Math.max(-d, 0.0));
            for (int k = i+1; k < dimension; k++) {
                L[i][k] = 0.0;
            }
        }
    }

    /** Solves the linear system of equations \(A\boldsymbol{x}=\boldsymbol{b}\) for \(\boldsymbol{x}\) and returns the
     * result as a new vector. The solution is obtained efficiently by using the Cholesky factor, \(L\). */
    public Vector solve(Vector vector) {
        return new Vector(solve(vector.copyAsMatrix()).getColumnPackedArrayCopy());
    }

    /** Solves the linear system of equations \(AX=B\) for \(X\) and returns the result as a new matrix. The solution is
     * obtained efficiently by using the Cholesky factor, \(L\). */
    public Matrix solve(Matrix matrix) {
        if (matrix.getRowDimension() != dimension) {
            throw new IllegalArgumentException("Matrix row dimensions must agree.");
        }
        if (!isSymmetricAndPositiveDefinite) {
            throw new RuntimeException("Matrix is not symmetric positive definite.");
        }
        double[][] rightHandSideMatrixArray = matrix.getArrayCopy();
        int resultMatrixColumnDimension = matrix.getColumnDimension();

        // Forward substitution solution
        for (int k = 0; k < dimension; k++) {
            for (int j = 0; j < resultMatrixColumnDimension; j++) {
                for (int i = 0; i < k ; i++) {
                    rightHandSideMatrixArray[k][j] -= rightHandSideMatrixArray[i][j] * L[k][i];
                }
                rightHandSideMatrixArray[k][j] /= L[k][k];
            }
        }

        // Backward substitution solution
        for (int k = dimension - 1; k >= 0; k--) {
            for (int j = 0; j < resultMatrixColumnDimension; j++) {
                for (int i = k+1; i < dimension; i++) {
                    rightHandSideMatrixArray[k][j] -= rightHandSideMatrixArray[i][j] * L[i][k];
                }
                rightHandSideMatrixArray[k][j] /= L[k][k];
            }
        }

        return new Matrix(rightHandSideMatrixArray, dimension, resultMatrixColumnDimension);
    }

    /**
     * Gets the Cholesky factor, \(L\).
     *
     * @return  The Cholesky factor, \(L\).
     */
    public Matrix getL() {
        return new Matrix(L, dimension, dimension);
    }

    /**
     * Gets the boolean value indicating whether or not the matrix whose decomposition is being computed (that is,
     * \(A\)) is symmetric and positive definite.
     *
     * @return  A boolean value indicating whether or not the matrix whose decomposition is being computed (that is,
     * \(A\)) is symmetric and positive definite.
     */
    public boolean isSymmetricAndPositiveDefinite() {
        return isSymmetricAndPositiveDefinite;
    }
}
