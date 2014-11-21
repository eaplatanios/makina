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

    /** A boolean value indicating whether the matrix whose decomposition is computed is symmetric. */
    private boolean isSymmetric;
    /** A boolean value indicating whether the matrix whose decomposition is computed is positive definite. */
    private boolean isPositiveDefinite;

    /**
     * Constructs a Cholesky decomposition object for the provided matrix. The actual decomposition is computed within
     * this constructor.
     *
     * @param   matrix  The matrix whose Cholesky decomposition is being computed.
     */
    public CholeskyDecomposition(Matrix matrix) {
        dimension = matrix.getRowDimension();
        isSymmetric = matrix.getColumnDimension() == dimension;
        isPositiveDefinite = isSymmetric;
        if (!isSymmetric) {
            throw new IllegalArgumentException("The matrix has to be square.");
        }
        double[][] matrixArray = matrix.getArray();
        L = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            double diagonalEntrySquare = 0.0;
            for (int j = 0; j < i; j++) {
                double temporarySum = 0.0;
                for (int k = 0; k < j; k++) {
                    temporarySum += L[j][k] * L[i][k];
                }
                L[i][j] = (matrixArray[i][j] - temporarySum)/L[j][j];
                diagonalEntrySquare += L[i][j] * L[i][j];
                isSymmetric &= (matrixArray[j][i] == matrixArray[i][j]);
            }
            diagonalEntrySquare -= matrixArray[i][i];
            isPositiveDefinite &= (diagonalEntrySquare < 0.0);
            L[i][i] = Math.sqrt(Math.max(-diagonalEntrySquare, 0.0));
            for (int k = i+1; k < dimension; k++) {
                L[i][k] = 0.0;
            }
        }
    }

    /**
     * Solves the linear system of equations \(A\boldsymbol{x}=\boldsymbol{b}\) for \(\boldsymbol{x}\) and returns the
     * result as a new vector. The solution is obtained efficiently by using the Cholesky factor, \(L\).
     *
     * @param   vector  Vector \(\boldsymbol{b}\) in equation \(A\boldsymbol{x}=\boldsymbol{b}\).
     * @return          The solution of the system of equations.
     */
    public Vector solve(Vector vector) throws NonSymmetricMatrixException, NonPositiveDefiniteMatrixException {
        if (vector.size() != dimension) {
            throw new IllegalArgumentException("Matrix row dimensions must agree.");
        }
        if (!isSymmetric) {
            throw new NonSymmetricMatrixException(
                    "Non symmetric matrix! A solution cannot be obtained using the Cholesky decomposition!"
            );
        }
        if (!isPositiveDefinite) {
            throw new NonPositiveDefiniteMatrixException(
                    "Non positive definite matrix! A solution cannot be obtained using the Cholesky decomposition!"
            );
        }
        Vector resultVector = vector.copy();
        // Forward substitution solution.
        for (int k = 0; k < dimension; k++) {
            for (int i = 0; i < k ; i++) {
                resultVector.set(k, resultVector.get(k) - resultVector.get(i) * L[k][i]);
            }
            resultVector.set(k, resultVector.get(k) / L[k][k]);
        }
        // Backward substitution solution.
        for (int k = dimension - 1; k >= 0; k--) {
            for (int i = k + 1; i < dimension; i++) {
                resultVector.set(k, resultVector.get(k) - resultVector.get(i) * L[i][k]);
            }
            resultVector.set(k, resultVector.get(k) / L[k][k]);
        }
        return resultVector;
    }

    /**
     * Solves the linear system of equations \(AX=B\) for \(X\) and returns the result as a new matrix. The solution is
     * obtained efficiently by using the Cholesky factor, \(L\).
     *
     * @param   matrix  Matrix \(B\) in equation \(AX=B\).
     * @return          The solution of the system of linear equations.
     */
    public Matrix solve(Matrix matrix) throws NonSymmetricMatrixException, NonPositiveDefiniteMatrixException {
        if (matrix.getRowDimension() != dimension) {
            throw new IllegalArgumentException("Matrix row dimensions must agree.");
        }
        if (!isSymmetric) {
            throw new NonSymmetricMatrixException(
                    "Non symmetric matrix! A solution cannot be obtained using the Cholesky decomposition!"
            );
        }
        if (!isPositiveDefinite) {
            throw new NonPositiveDefiniteMatrixException(
                    "Non positive definite matrix! A solution cannot be obtained using the Cholesky decomposition!"
            );
        }
        double[][] rightHandSideMatrixArray = matrix.getArrayCopy();
        int resultMatrixColumnDimension = matrix.getColumnDimension();
        // Forward substitution solution.
        for (int k = 0; k < dimension; k++) {
            for (int j = 0; j < resultMatrixColumnDimension; j++) {
                for (int i = 0; i < k ; i++) {
                    rightHandSideMatrixArray[k][j] -= rightHandSideMatrixArray[i][j] * L[k][i];
                }
                rightHandSideMatrixArray[k][j] /= L[k][k];
            }
        }
        // Backward substitution solution.
        for (int k = dimension - 1; k >= 0; k--) {
            for (int j = 0; j < resultMatrixColumnDimension; j++) {
                for (int i = k + 1; i < dimension; i++) {
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
     * @return  The Cholesky factor, \(L\), as a new matrix.
     */
    public Matrix getL() {
        return new Matrix(L, dimension, dimension);
    }

    /**
     * Gets the boolean value indicating whether or not the matrix whose decomposition is being computed (that is,
     * \(A\)) is symmetric.
     *
     * @return  A boolean value indicating whether or not the matrix whose decomposition is being computed (that is,
     * \(A\)) is symmetric.
     */
    public boolean isSymmetric() {
        return isSymmetric;
    }

    /**
     * Gets the boolean value indicating whether or not the matrix whose decomposition is being computed (that is,
     * \(A\)) is positive definite.
     *
     * @return  A boolean value indicating whether or not the matrix whose decomposition is being computed (that is,
     * \(A\)) is positive definite.
     */
    public boolean isPositiveDefinite() {
        return isPositiveDefinite;
    }

    /**
     * Gets the boolean value indicating whether or not the matrix whose decomposition is being computed (that is,
     * \(A\)) is symmetric and positive definite.
     *
     * @return  A boolean value indicating whether or not the matrix whose decomposition is being computed (that is,
     * \(A\)) is symmetric and positive definite.
     */
    public boolean isSymmetricAndPositiveDefinite() {
        return isSymmetric && isPositiveDefinite;
    }
}
