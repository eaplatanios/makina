package org.platanios.learn.math.matrix;

import java.util.Arrays;

/**
 * Implements the LU decomposition algorithm for matrix \(A\). Given \(A\in\mathbb{R}^{m\times n}\), with \(m\geq n\),
 * the LU decomposition is an upper triangular matrix \(U\in\mathbb{R}^{n\times n}\), a unit lower triangular matrix
 * \(L\in\mathbb{R}^{m\times n}\) and a permutation/pivot vector \(\boldsymbol{p}\in\mathbb{R}^m\), such that
 * \(A(\boldsymbol{p},:)=LU\) (the permutation/pivot vector in this equation can be thought of as re-ordering the rows
 * of \(A\)). If \(m&lt;n\), then \(L\in\mathbb{R}^{m\times m}\) and \(U\in\mathbb{R}^{m\times n}\). The LU decomposition
 * with pivoting always exists, even if the matrix is singular, and so the constructor of this class will never fail.
 * Furthermore, the primary use of the LU decomposition is in the solution of square systems of simultaneous matrix
 * equations. In that case, in order to obtain a solution, the matrix has to be non-singular.
 *
 * @author Emmanouil Antonios Platanios
 */
public class LUDecomposition {
    /** Two-dimensional array used for internal storage of the decomposition factors. */
    private final double[][] LU;
    /** The row dimension of the matrix whose decomposition is being computed. */
    private final int rowDimension;
    /** The column dimension of the matrix whose decomposition is being computed. */
    private final int columnDimension;
    /** One-dimensional array used for internal storage of the pivot vector. */
    private final int[] pivot;

    /** An integer holding the pivot sign. Its value is {@code 1} for positive sign and {@code -1} for negative sign. */
    private int pivotSign;
    /** A boolean value indicating whether or not the matrix whose decomposition is being computed is non-singular. */
    private boolean isNonSingular;

    /**
     * Constructs an LU decomposition object for the provided matrix. The actual decomposition is computed within this
     * constructor using Crout's algorithm.
     *
     * @param   matrix  The matrix whose LU decomposition is being computed.
     */
    public LUDecomposition(Matrix matrix) {
        LU = matrix.getArrayCopy();
        rowDimension = matrix.getRowDimension();
        columnDimension = matrix.getColumnDimension();
        pivot = new int[rowDimension];
        for (int i = 0; i < rowDimension; i++) {
            pivot[i] = i;
        }
        pivotSign = 1;
        isNonSingular = true;

        for (int j = 0; j < columnDimension; j++) {
            // Apply previous transformations.
            for (int i = 0; i < rowDimension; i++) {
                int maximumDimension = Math.min(i, j);
                double temporarySum = 0.0;
                for (int k = 0; k < maximumDimension; k++) {
                    temporarySum += LU[i][k] * LU[k][j];
                }
                LU[i][j] -= temporarySum;
                LU[i][j] = LU[i][j];
            }
            // Find pivot and exchange if necessary.
            int p = j;
            for (int i = j + 1; i < rowDimension; i++) {
                if (Math.abs(LU[i][j]) > Math.abs(LU[p][j])) {
                    p = i;
                }
            }
            if (p != j) {
                for (int k = 0; k < columnDimension; k++) {
                    double temporaryExchangeVariable = LU[p][k];
                    LU[p][k] = LU[j][k];
                    LU[j][k] = temporaryExchangeVariable;
                }
                int temporaryExchangeVariable = pivot[p];
                pivot[p] = pivot[j];
                pivot[j] = temporaryExchangeVariable;
                pivotSign = -pivotSign;
            }
            // Compute the multipliers.
            if (j < rowDimension & LU[j][j] != 0.0) {
                for (int i = j + 1; i < rowDimension; i++) {
                    LU[i][j] /= LU[j][j];
                }
            }
            if (LU[j][j] == 0) {
                isNonSingular = false;
            }
        }
    }

    /**
     * Solves the linear system of equations \(A\boldsymbol{x}=\boldsymbol{b}\) for \(\boldsymbol{x}\) and returns the
     * result as a new vector. The solution is obtained efficiently by using the LU decomposition.
     *
     * @param   vector  Vector \(\boldsymbol{b}\) in equation \(A\boldsymbol{x}=\boldsymbol{b}\).
     * @return          The solution of the system of equations.
     */
    public Vector solve(Vector vector) throws SingularMatrixException {
        return new Vector(solve(vector.copyAsMatrix()).getColumnPackedArrayCopy());
    }

    /**
     * Solves the linear system of equations \(AX=B\) for \(X\) and returns the result as a new matrix. The solution is
     * obtained efficiently by using the LU decomposition.
     *
     * @param   matrix  Matrix \(B\) in equation \(AX=B\).
     * @return          The solution of the system of linear equations.
     */
    public Matrix solve(Matrix matrix) throws SingularMatrixException {
        if (matrix.getRowDimension() != rowDimension) {
            throw new IllegalArgumentException("Matrix row dimensions must agree.");
        }
        if (!isNonSingular) {
            throw new SingularMatrixException(
                    "Singular matrix! A solution cannot be obtained using the LU decomposition!"
            );
        }
        int resultMatrixColumnDimension = matrix.getColumnDimension();
        Matrix resultMatrix = matrix.getSubMatrix(pivot, 0, resultMatrixColumnDimension - 1);
        double[][] resultMatrixArray = resultMatrix.getArray();
        // Solve \(LY=B(pivot,:)\).
        for (int k = 0; k < columnDimension; k++) {
            for (int i = k + 1; i < columnDimension; i++) {
                for (int j = 0; j < resultMatrixColumnDimension; j++) {
                    resultMatrixArray[i][j] -= resultMatrixArray[k][j] * LU[i][k];
                }
            }
        }
        // Solve \(UX=Y\).
        for (int k = columnDimension - 1; k >= 0; k--) {
            for (int j = 0; j < resultMatrixColumnDimension; j++) {
                resultMatrixArray[k][j] /= LU[k][k];
            }
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < resultMatrixColumnDimension; j++) {
                    resultMatrixArray[i][j] -= resultMatrixArray[k][j] * LU[i][k];
                }
            }
        }
        return resultMatrix;
    }

    /**
     * Gets the lower triangular factor, \(L\).
     *
     * @return  The lower triangular factor, \(L\), as a new matrix.
     */
    public Matrix getL() {
        Matrix L = new Matrix(rowDimension, columnDimension);
        double[][] lArray = L.getArray();
        for (int i = 0; i < rowDimension; i++) {
            for (int j = 0; j <= i; j++) {
                if (i == j) {
                    lArray[i][j] = 1.0;
                } else {
                    lArray[i][j] = LU[i][j];
                }
            }
        }
        return L;
    }

    /**
     * Gets the upper triangular factor, \(U\).
     *
     * @return  The upper triangular factor, \(U\), as a new matrix.
     */
    public Matrix getU() {
        Matrix U = new Matrix(rowDimension, columnDimension);
        double[][] uArray = U.getArray();
        for (int i = 0; i < columnDimension; i++) {
            System.arraycopy(LU[i], i, uArray[i], i, columnDimension - i);
        }
        return U;
    }

    /**
     * Gets the pivot (permutation) vector represented as a new array.
     *
     * @return  The pivot vector as a new integer array.
     */
    public int[] getPivot() {
        return Arrays.copyOf(pivot, pivot.length);
    }

    /**
     * Computes the determinant of the matrix whose decomposition is being computed.
     *
     * @return  The determinant of the matrix whose decomposition is being computed.
     */
    public double computeDeterminant() {
        if (rowDimension != columnDimension) {
            throw new IllegalArgumentException("Matrix must be square.");
        }
        double determinant = (double) pivotSign;
        for (int j = 0; j < columnDimension; j++) {
            determinant *= LU[j][j];
        }
        return determinant;
    }

    /**
     * Gets the boolean value indicating whether or not the matrix whose decomposition is being computed is
     * non-singular.
     *
     * @return  A boolean value indicating whether or not the matrix whose decomposition is being computed is
     *          non-singular.
     */
    public boolean isNonSingular() {
        return isNonSingular;
    }
}
