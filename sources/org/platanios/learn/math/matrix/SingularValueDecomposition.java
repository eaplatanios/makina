package org.platanios.learn.math.matrix;

import org.platanios.learn.math.Utilities;

import java.util.Arrays;

/**
 * Implements the singular value decomposition (SVD) algorithm for matrix \(A\). Given \(A\in\mathbb{R}^{m\times n}\),
 * with \(m\geq n\), the singular value decomposition is an orthogonal matrix \(U\in\mathbb{R}^{m\times n}\), a diagonal
 * matrix \(S\in\mathbb{R}^{n\times n}\) and an orthogonal matrix \(V\in\mathbb{R}^{n\times n}\), such that
 * \(A=USV^T\). The diagonal of \(S\) contains the singular values of the matrix \(A\) and these values are ordered such
 * that: \(\sigma_0\geq\sigma_1\geq\hdots\geq\sigma_{n-1}\). The singular value decomposition always exists.
 * Furthermore, the condition number, the effective numerical rank and the pseudo-inverse of the matrix \(A\) can be
 * computed using this decomposition.
 *
 * @author Emmanouil Antonios Platanios
 */
public class SingularValueDecomposition {
    /** Relative threshold for small singular values. */
    private static final double RELATIVE_SMALL_SINGULAR_VALUES_THRESHOLD = 0x1.0p-52;
    /** Absolute threshold for small singular values. */
    private static final double ABSOLUTE_SMALL_SINGULAR_VALUES_THRESHOLD = 0x1.0p-966;

    /** Two-dimensional array used for internal storage of the \(U\) decomposition factor. */
    private final double[][] U;
    /** Two-dimensional array used for internal storage of the \(V\) decomposition factor. */
    private final double[][] V;
    /** One-dimensional array used for internal storage of the diagonal of the \(S\) decomposition factor. */
    private final double[] singularValues;
    /** The row dimension of the matrix whose decomposition is being computed. */
    private final int rowDimension;
    /** The column dimension of the matrix whose decomposition is being computed. */
    private final int columnDimension;
    /** The tolerance for small singular values. */
    private final double singularValuesTolerance;

    /**
     * Constructs an singular value decomposition object for the provided matrix. This implementation is derived from
     * the LINPACK code.
     *
     * @param   matrix  The matrix whose singular value decomposition is being computed.
     */
    public SingularValueDecomposition(Matrix matrix) {
        final double[][] matrixArray;
        final double[][] tempU;
        final double[][] tempV;
        final boolean transposed;
        // Transpose if necessary in order to make sure that the number of rows is greater than or equal to the number
        // of columns.
        if (matrix.getRowDimension() < matrix.getColumnDimension()) {
            transposed = true;
            matrixArray = matrix.transpose().getArrayCopy();
            rowDimension = matrix.getColumnDimension();
            columnDimension = matrix.getRowDimension();
        } else {
            transposed = false;
            matrixArray = matrix.getArrayCopy();
            rowDimension = matrix.getRowDimension();
            columnDimension = matrix.getColumnDimension();
        }
        singularValues = new double[columnDimension];
        tempU = new double[rowDimension][columnDimension];
        tempV = new double[columnDimension][columnDimension];
        final double[] e = new double[columnDimension];
        final double[] work = new double[rowDimension];
        // Reduce the matrix to bi-diagonal form, storing the diagonal elements in the singular values array and the
        // super-diagonal elements in array e.
        final int nct = Math.min(rowDimension - 1, columnDimension);
        final int nrt = Math.max(0, columnDimension - 2);
        for (int k = 0; k < Math.max(nct, nrt); k++) {
            if (k < nct) {
                // Compute the transformation for the k-th column and place the k-th diagonal in s[k].
                singularValues[k] = 0;
                for (int i = k; i < rowDimension; i++) {
                    singularValues[k] = Utilities.computeHypotenuse(singularValues[k], matrixArray[i][k]);
                }
                if (singularValues[k] != 0) {
                    if (matrixArray[k][k] < 0) {
                        singularValues[k] = -singularValues[k];
                    }
                    for (int i = k; i < rowDimension; i++) {
                        matrixArray[i][k] /= singularValues[k];
                    }
                    matrixArray[k][k] += 1;
                }
                singularValues[k] = -singularValues[k];
            }
            for (int j = k + 1; j < columnDimension; j++) {
                if (k < nct && singularValues[k] != 0) {
                    // Apply the transformation.
                    double temporarySum = 0;
                    for (int i = k; i < rowDimension; i++) {
                        temporarySum -= matrixArray[i][k] * matrixArray[i][j];
                    }
                    temporarySum /= matrixArray[k][k];
                    for (int i = k; i < rowDimension; i++) {
                        matrixArray[i][j] += temporarySum * matrixArray[i][k];
                    }
                }
                // Place the k-th row of A into array e for the subsequent calculation of the row transformation.
                e[j] = matrixArray[k][j];
            }
            if (k < nct) {
                // Place the transformation in U for the subsequent back multiplication.
                for (int i = k; i < rowDimension; i++) {
                    tempU[i][k] = matrixArray[i][k];
                }
            }
            if (k < nrt) {
                // Compute the k-th row transformation and place the k-th super-diagonal in e[k].
                e[k] = 0;
                for (int i = k + 1; i < columnDimension; i++) {
                    e[k] = Utilities.computeHypotenuse(e[k], e[i]);
                }
                if (e[k] != 0) {
                    if (e[k + 1] < 0) {
                        e[k] = -e[k];
                    }
                    for (int i = k + 1; i < columnDimension; i++) {
                        e[i] /= e[k];
                    }
                    e[k + 1] += 1;
                }
                e[k] = -e[k];
                if (k + 1 < rowDimension && e[k] != 0) {
                    // Apply the transformation.
                    for (int i = k + 1; i < rowDimension; i++) {
                        work[i] = 0;
                    }
                    for (int j = k + 1; j < columnDimension; j++) {
                        for (int i = k + 1; i < rowDimension; i++) {
                            work[i] += e[j] * matrixArray[i][j];
                        }
                    }
                    for (int j = k + 1; j < columnDimension; j++) {
                        final double t = -e[j] / e[k + 1];
                        for (int i = k + 1; i < rowDimension; i++) {
                            matrixArray[i][j] += t * work[i];
                        }
                    }
                }
                // Place the transformation in V for the subsequent back multiplication.
                for (int i = k + 1; i < columnDimension; i++) {
                    tempV[i][k] = e[i];
                }
            }
        }
        // Set up the final bi-diagonal matrix.
        int p = columnDimension;
        if (nct < columnDimension) {
            singularValues[nct] = matrixArray[nct][nct];
        }
        if (rowDimension < p) {
            singularValues[p - 1] = 0;
        }
        if (nrt + 1 < p) {
            e[nrt] = matrixArray[nrt][p - 1];
        }
        e[p - 1] = 0;
        // Generate U.
        for (int j = nct; j < columnDimension; j++) {
            for (int i = 0; i < rowDimension; i++) {
                tempU[i][j] = 0;
            }
            tempU[j][j] = 1;
        }
        for (int k = nct - 1; k >= 0; k--) {
            if (singularValues[k] != 0) {
                for (int j = k + 1; j < columnDimension; j++) {
                    double temporarySum = 0;
                    for (int i = k; i < rowDimension; i++) {
                        temporarySum -= tempU[i][k] * tempU[i][j];
                    }
                    temporarySum /= tempU[k][k];
                    for (int i = k; i < rowDimension; i++) {
                        tempU[i][j] += temporarySum * tempU[i][k];
                    }
                }
                for (int i = k; i < rowDimension; i++) {
                    tempU[i][k] = -tempU[i][k];
                }
                tempU[k][k] = 1 + tempU[k][k];
                for (int i = 0; i < k - 1; i++) {
                    tempU[i][k] = 0;
                }
            } else {
                for (int i = 0; i < rowDimension; i++) {
                    tempU[i][k] = 0;
                }
                tempU[k][k] = 1;
            }
        }
        // Generate V.
        for (int k = columnDimension - 1; k >= 0; k--) {
            if (k < nrt && e[k] != 0) {
                for (int j = k + 1; j < columnDimension; j++) {
                    double temporarySum = 0;
                    for (int i = k + 1; i < columnDimension; i++) {
                        temporarySum -= tempV[i][k] * tempV[i][j];
                    }
                    temporarySum /= tempV[k + 1][k];
                    for (int i = k + 1; i < columnDimension; i++) {
                        tempV[i][j] += temporarySum * tempV[i][k];
                    }
                }
            }
            for (int i = 0; i < columnDimension; i++) {
                tempV[i][k] = 0;
            }
            tempV[k][k] = 1;
        }
        // Main loop for generating the singular values.
        final int pp = p - 1;
        int iterationNumber = 0;
        while (p > 0) {
            int k;
            int caseNumber;
            // This section of the program inspects for negligible elements in the s and e arrays. On completion the
            // variables caseNumber and k are set as follows.
            // caseNumber = 1: If s[p] and e[k-1] are negligible and k < p.
            // caseNumber = 2: If s[k] is negligible and k < p.
            // caseNumber = 3: If e[k-1] is negligible, k < p, and s[k],...,s[p] are not negligible (perform QR step).
            // caseNumber = 4: If e[p-1] is negligible (that is, we have convergence).
            for (k = p - 2; k >= 0; k--) {
                final double threshold = ABSOLUTE_SMALL_SINGULAR_VALUES_THRESHOLD
                        + RELATIVE_SMALL_SINGULAR_VALUES_THRESHOLD * (Math.abs(singularValues[k])
                        + Math.abs(singularValues[k + 1]));
                // This condition is written in this way so that we break out of loop if NaN values are encountered.
                if (!(Math.abs(e[k]) > threshold)) {
                    e[k] = 0;
                    break;
                }
            }
            if (k == p - 2) {
                caseNumber = 4;
            } else {
                int ks;
                for (ks = p - 1; ks >= k; ks--) {
                    if (ks == k) {
                        break;
                    }
                    final double t = (ks != p ? Math.abs(e[ks]) : 0) + (ks != k + 1 ? Math.abs(e[ks - 1]) : 0);
                    if (Math.abs(singularValues[ks]) <=
                            ABSOLUTE_SMALL_SINGULAR_VALUES_THRESHOLD + RELATIVE_SMALL_SINGULAR_VALUES_THRESHOLD * t) {
                        singularValues[ks] = 0;
                        break;
                    }
                }
                if (ks == k) {
                    caseNumber = 3;
                } else if (ks == p - 1) {
                    caseNumber = 1;
                } else {
                    caseNumber = 2;
                    k = ks;
                }
            }
            k++;
            double f;
            // Perform the task indicated by caseNumber.
            switch (caseNumber) {
                case 1: // Deflate negligible s[p] value.
                    f = e[p - 2];
                    e[p - 2] = 0;
                    for (int j = p - 2; j >= k; j--) {
                        double t = Utilities.computeHypotenuse(singularValues[j], f);
                        final double cs = singularValues[j] / t;
                        final double sn = f / t;
                        singularValues[j] = t;
                        if (j != k) {
                            f = -sn * e[j - 1];
                            e[j - 1] = cs * e[j - 1];
                        }
                        for (int i = 0; i < columnDimension; i++) {
                            t = cs * tempV[i][j] + sn * tempV[i][p - 1];
                            tempV[i][p - 1] = -sn * tempV[i][j] + cs * tempV[i][p - 1];
                            tempV[i][j] = t;
                        }
                    }
                    break;
                case 2: // Split at negligible s[k] value.
                    f = e[k - 1];
                    e[k - 1] = 0;
                    for (int j = k; j < p; j++) {
                        double t = Utilities.computeHypotenuse(singularValues[j], f);
                        final double cs = singularValues[j] / t;
                        final double sn = f / t;
                        singularValues[j] = t;
                        f = -sn * e[j];
                        e[j] = cs * e[j];
                        for (int i = 0; i < rowDimension; i++) {
                            t = cs * tempU[i][j] + sn * tempU[i][k - 1];
                            tempU[i][k - 1] = -sn * tempU[i][j] + cs * tempU[i][k - 1];
                            tempU[i][j] = t;
                        }
                    }
                    break;
                case 3: // Perform one QR step.
                    // Calculate the shift.
                    final double maxPm1Pm2 = Math.max(Math.abs(singularValues[p - 1]), Math.abs(singularValues[p - 2]));
                    final double scale = Math.max(
                            Math.max(
                                    Math.max(
                                            maxPm1Pm2,
                                            Math.abs(e[p - 2])
                                    ),
                                    Math.abs(singularValues[k])
                            ),
                            Math.abs(e[k])
                    );
                    final double sp = singularValues[p - 1] / scale;
                    final double spm1 = singularValues[p - 2] / scale;
                    final double epm1 = e[p - 2] / scale;
                    final double sk = singularValues[k] / scale;
                    final double ek = e[k] / scale;
                    final double b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
                    final double c = (sp * epm1) * (sp * epm1);
                    double shift = 0;
                    if (b != 0 || c != 0) {
                        shift = Math.sqrt(b * b + c);
                        if (b < 0) {
                            shift = -shift;
                        }
                        shift = c / (b + shift);
                    }
                    f = (sk + sp) * (sk - sp) + shift;
                    double g = sk * ek;
                    // Chase zeros.
                    for (int j = k; j < p - 1; j++) {
                        double t = Utilities.computeHypotenuse(f, g);
                        double cs = f / t;
                        double sn = g / t;
                        if (j != k) {
                            e[j - 1] = t;
                        }
                        f = cs * singularValues[j] + sn * e[j];
                        e[j] = cs * e[j] - sn * singularValues[j];
                        g = sn * singularValues[j + 1];
                        singularValues[j + 1] = cs * singularValues[j + 1];
                        for (int i = 0; i < columnDimension; i++) {
                            t = cs * tempV[i][j] + sn * tempV[i][j + 1];
                            tempV[i][j + 1] = -sn * tempV[i][j] + cs * tempV[i][j + 1];
                            tempV[i][j] = t;
                        }
                        t = Utilities.computeHypotenuse(f, g);
                        cs = f / t;
                        sn = g / t;
                        singularValues[j] = t;
                        f = cs * e[j] + sn * singularValues[j + 1];
                        singularValues[j + 1] = -sn * e[j] + cs * singularValues[j + 1];
                        g = sn * e[j + 1];
                        e[j + 1] = cs * e[j + 1];
                        if (j < rowDimension - 1) {
                            for (int i = 0; i < rowDimension; i++) {
                                t = cs * tempU[i][j] + sn * tempU[i][j + 1];
                                tempU[i][j + 1] = -sn * tempU[i][j] + cs * tempU[i][j + 1];
                                tempU[i][j] = t;
                            }
                        }
                    }
                    e[p - 2] = f;
                    iterationNumber++;
                    break;
                default: // Convergence.
                    // Make the singular values positive.
                    if (singularValues[k] <= 0) {
                        singularValues[k] = singularValues[k] < 0 ? -singularValues[k] : 0;
                        for (int i = 0; i <= pp; i++) {
                            tempV[i][k] = -tempV[i][k];
                        }
                    }
                    // Order the singular values.
                    while (k < pp) {
                        if (singularValues[k] >= singularValues[k + 1]) {
                            break;
                        }
                        double t = singularValues[k];
                        singularValues[k] = singularValues[k + 1];
                        singularValues[k + 1] = t;
                        if (k < columnDimension - 1) {
                            for (int i = 0; i < columnDimension; i++) {
                                t = tempV[i][k + 1];
                                tempV[i][k + 1] = tempV[i][k];
                                tempV[i][k] = t;
                            }
                        }
                        if (k < rowDimension - 1) {
                            for (int i = 0; i < rowDimension; i++) {
                                t = tempU[i][k + 1];
                                tempU[i][k + 1] = tempU[i][k];
                                tempU[i][k] = t;
                            }
                        }
                        k++;
                    }
                    iterationNumber = 0;
                    p--;
                    break;
            }
        }
        // Set the tolerance for small singular values, used to calculate the rank and the pseudo-inverse of the matrix
        // whose decomposition is being computed.
        singularValuesTolerance = Math.max(rowDimension * singularValues[0] * RELATIVE_SMALL_SINGULAR_VALUES_THRESHOLD,
                                          Math.sqrt(Utilities.computeMachineEpsilonDouble()));
        if (!transposed) {
            U = tempU;
            V = tempV;
        } else {
            U = tempV;
            V = tempU;
        }
    }

    /**
     * Solves the linear system of equations \(A\boldsymbol{x}=\boldsymbol{b}\) for \(\boldsymbol{x}\) and returns the
     * result as a new vector. The solution is obtained using the pseudoinverse which is in turn computed using the
     * singular value decomposition.
     *
     * @param   vector  Vector \(\boldsymbol{b}\) in equation \(A\boldsymbol{x}=\boldsymbol{b}\).
     * @return          The solution of the system of equations.
     */
    public Vector solve(Vector vector) {
        return new Vector(solve(vector.copyAsMatrix()).getColumnPackedArrayCopy());
    }

    /**
     * Solves the linear system of equations \(AX=B\) for \(X\) and returns the result as a new matrix. The solution is
     * obtained using the pseudoinverse which is in turn computed using the singular value decomposition.
     *
     * @param   matrix  Matrix \(B\) in equation \(AX=B\).
     * @return          The solution of the system of linear equations.
     */
    public Matrix solve(Matrix matrix) {
        final double[][] uTransposeArray = (new Matrix(U)).transpose().getArrayCopy();
        for (int i = 0; i < singularValues.length; ++i) {
            final double adjustedSingularValue;
            if (singularValues[i] > singularValuesTolerance) {
                adjustedSingularValue = 1 / singularValues[i];
            } else {
                adjustedSingularValue = 0;
            }
            for (int j = 0; j < uTransposeArray[i].length; ++j) {
                uTransposeArray[i][j] *= adjustedSingularValue;
            }
        }
        Matrix matrixPseudoInverse = (new Matrix(V)).multiply(new Matrix(uTransposeArray, false));
        return matrixPseudoInverse.multiply(matrix);
    }

    /**
     * Gets the orthogonal factor, \(U\).
     *
     * @return  The orthogonal factor, \(U\), as a new matrix.
     */
    public Matrix getU() {
        return new Matrix(U, rowDimension, columnDimension);
    }

    /**
     * Gets the diagonal factor, \(S\). The diagonal entries of this matrix are the singular values of the matrix whose
     * decomposition is being computed.
     *
     * @return  The diagonal factor, \(S\), as a new matrix.
     */
    public Matrix getS() {
        Matrix S = new Matrix(columnDimension, columnDimension);
        double[][] sArray = S.getArray();
        for (int i = 0; i < columnDimension; i++) {
            sArray[i][i] = singularValues[i];
        }
        return S;
    }

    /**
     * Gets the singular values of the matrix whose decomposition is being computed.
     *
     * @return  A one-dimensional array containing the singular values of the matrix whose decomposition is being
     *          computed.
     */
    public double[] getSingularValues() {
        return Arrays.copyOf(singularValues, singularValues.length);
    }

    /**
     * Gets the orthogonal factor, \(V\).
     *
     * @return  The orthogonal factor, \(V\), as a new matrix.
     */
    public Matrix getV() {
        return new Matrix(V, columnDimension, columnDimension);
    }

    /**
     * Computes the L2 norm of the matrix whose decomposition is being computed.
     *
     * @return  The L2 norm of the matrix whose decomposition is being computed.
     */
    public double computeL2Norm() {
        return singularValues[0];
    }

    /**
     * Computes the condition number of the matrix whose decomposition is being computed.
     *
     * @return  The condition number of the matrix whose decomposition is being computed.
     */
    public double computeConditionNumber() {
        return singularValues[0] / singularValues[columnDimension - 1];
    }

    /**
     * Computes the effective numerical rank of the matrix whose decomposition is being computed. The effective
     * numerical rank is the number of non-negligible singular values. The threshold used to identify non-negligible
     * terms is equal to \(\max{m,n}\text{lsb}(s_1)\) where \(\text{lsb}(s_1)\) is the least significant bit of the
     * largest singular value.
     *
     * @return  The effective numerical rank of the matrix whose decomposition is being computed.
     */
    public int computeEffectiveNumericalRank() {
        int effectiveNumericalRank = 0;
        for (double singularValue : singularValues) {
            if (singularValue > singularValuesTolerance) {
                effectiveNumericalRank++;
            }
        }
        return effectiveNumericalRank;
    }
}
