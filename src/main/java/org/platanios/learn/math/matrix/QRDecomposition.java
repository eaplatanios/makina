package org.platanios.learn.math.matrix;

/**
 * Implements the QR decomposition algorithm for matrix \(A\). Given \(A\in\mathbb{R}^{m\times n}\), with \(m\geq n\),
 * the QR decomposition is an orthogonal matrix \(Q\in\mathbb{R}^{m\times n}\) and an upper triangular matrix
 * \(R\in\mathbb{R}^{n\times n}\), such that \(A=QR\). The QR decomposition always exists, even if the matrix does not
 * have full rank, and so the constructor of this class will never fail. Furthermore, the primary use of the QR
 * decomposition is in the least squares solution of non-square systems of simultaneous matrix equations. In that case,
 * in order to obtain a solution, the matrix has to have full rank.
 *
 * @author Emmanouil Antonios Platanios
 */
public class QRDecomposition {
    /** Two-dimensional array used for internal storage of the decomposition factors. */
    private final double[][] QR;
    /** One-dimensional array used for internal storage of the diagonal of the upper triangular factor, \(R\). */
    private final double[] rDiagonal;
    /** The row dimension of the matrix whose decomposition is being computed. */
    private final int rowDimension;
    /** The column dimension of the matrix whose decomposition is being computed. */
    private final int columnDimension;

    /** A boolean value indicating whether or not the matrix whose decomposition is being computed is full rank. */
    private boolean isFullRank;

    /**
     * Constructs a QR decomposition object for the provided matrix. The actual decomposition is computed within this
     * constructor using Householder reflections.
     *
     * @param   matrix  The matrix whose QR decomposition is being computed.
     */
    public QRDecomposition(Matrix matrix) {
        QR = matrix.getArrayCopy();
        rowDimension = matrix.getRowDimension();
        columnDimension = matrix.getColumnDimension();
        rDiagonal = new double[columnDimension];
        isFullRank = true;

        for (int k = 0; k < columnDimension; k++) {
            double columnL2Norm = 0;
            for (int i = k; i < rowDimension; i++) {
                columnL2Norm = org.platanios.learn.math.Utilities.computeHypotenuse(columnL2Norm, QR[i][k]);
            }
            if (columnL2Norm != 0.0) {
                if (QR[k][k] < 0) {
                    columnL2Norm = -columnL2Norm;
                }
                for (int i = k; i < rowDimension; i++) {
                    QR[i][k] /= columnL2Norm;
                }
                QR[k][k] += 1.0;
                for (int j = k + 1; j < columnDimension; j++) {
                    double temporarySum = 0.0;
                    for (int i = k; i < rowDimension; i++) {
                        temporarySum -= QR[i][k] * QR[i][j];
                    }
                    temporarySum /= QR[k][k];
                    for (int i = k; i < rowDimension; i++) {
                        QR[i][j] += temporarySum * QR[i][k];
                    }
                }
            }
            rDiagonal[k] = -columnL2Norm;
            if (rDiagonal[k] == 0) {
                isFullRank = false;
            }
        }
    }

    /**
     * Solves the linear system of equations \(A\boldsymbol{x}=\boldsymbol{b}\) for \(\boldsymbol{x}\) and returns the
     * result as a new vector. The solution is obtained efficiently by using the QR decomposition.
     *
     * @param   vector  Vector \(\boldsymbol{b}\) in equation \(A\boldsymbol{x}=\boldsymbol{b}\).
     * @return          The solution of the system of equations.
     */
    public Vector solve(Vector vector) throws SingularMatrixException {
        if (vector.size() != rowDimension) {
            throw new IllegalArgumentException("Matrix row dimensions must agree.");
        }
        if (!isFullRank) {
            throw new SingularMatrixException(
                    "Rank deficient matrix! A solution cannot be obtained using the QR decomposition!"
            );
        }
        Vector resultVector = vector.copy();
        // Compute \(Y=Q^TB\).
        for (int k = 0; k < columnDimension; k++) {
            double temporarySum = 0.0;
            for (int i = k; i < rowDimension; i++) {
                temporarySum -= QR[i][k] * resultVector.get(i);
            }
            temporarySum /= QR[k][k];
            for (int i = k; i < rowDimension; i++) {
                resultVector.set(i, resultVector.get(i) + temporarySum * QR[i][k]);
            }
        }
        // Solve \(RX=Y\).
        for (int k = columnDimension - 1; k >= 0; k--) {
            resultVector.set(k, resultVector.get(k) / rDiagonal[k]);
            for (int i = 0; i < k; i++) {
                resultVector.set(i, resultVector.get(i) - resultVector.get(k) * QR[i][k]);
            }
        }
        return resultVector.get(0, columnDimension - 1);
    }

    /**
     * Solves the linear system of equations \(AX=B\) for \(X\) and returns the result as a new matrix. The solution is
     * obtained efficiently by using the QR decomposition.
     *
     * @param   matrix  Matrix \(B\) in equation \(AX=B\).
     * @return          The solution of the system of linear equations.
     */
    public Matrix solve(Matrix matrix) throws SingularMatrixException {
        if (matrix.getRowDimension() != rowDimension) {
            throw new IllegalArgumentException("Matrix row dimensions must agree.");
        }
        if (!isFullRank) {
            throw new SingularMatrixException(
                    "Rank deficient matrix! A solution cannot be obtained using the QR decomposition!"
            );
        }
        double[][] rightHandSideMatrixArray = matrix.getArrayCopy();
        int resultMatrixColumnDimension = matrix.getColumnDimension();
        // Compute \(Y=Q^TB\).
        for (int k = 0; k < columnDimension; k++) {
            for (int j = 0; j < resultMatrixColumnDimension; j++) {
                double temporarySum = 0.0;
                for (int i = k; i < rowDimension; i++) {
                    temporarySum -= QR[i][k] * rightHandSideMatrixArray[i][j];
                }
                temporarySum /= QR[k][k];
                for (int i = k; i < rowDimension; i++) {
                    rightHandSideMatrixArray[i][j] += temporarySum * QR[i][k];
                }
            }
        }
        // Solve \(RX=Y\).
        for (int k = columnDimension - 1; k >= 0; k--) {
            for (int j = 0; j < resultMatrixColumnDimension; j++) {
                rightHandSideMatrixArray[k][j] /= rDiagonal[k];
            }
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < resultMatrixColumnDimension; j++) {
                    rightHandSideMatrixArray[i][j] -= rightHandSideMatrixArray[k][j] * QR[i][k];
                }
            }
        }
        return (new Matrix(rightHandSideMatrixArray, columnDimension, resultMatrixColumnDimension))
                .getSubMatrix(0, columnDimension - 1, 0, resultMatrixColumnDimension - 1);
    }

    /**
     * Gets the Householder reflection vectors.
     *
     * @return  A lower trapezoidal matrix whose columns define the reflections, as a new matrix.
     */
    public Matrix getH() {
        Matrix H = new Matrix(rowDimension, columnDimension);
        double[][] hArray = H.getArray();
        for (int i = 0; i < rowDimension; i++) {
            System.arraycopy(QR[i], 0, hArray[i], 0, i + 1);
        }
        return H;
    }

    /**
     * Gets the orthogonal factor, \(Q\).
     *
     * @return  The orthogonal factor, \(Q\), as a new matrix.
     */
    public Matrix getQ() {
        Matrix Q = new Matrix(rowDimension, columnDimension);
        double[][] qArray = Q.getArray();
        for (int k = columnDimension - 1; k >= 0; k--) {
            for (int i = 0; i < rowDimension; i++) {
                qArray[i][k] = 0.0;
            }
            qArray[k][k] = 1.0;
            for (int j = k; j < columnDimension; j++) {
                if (QR[k][k] != 0) {
                    double temporarySum = 0.0;
                    for (int i = k; i < rowDimension; i++) {
                        temporarySum -= QR[i][k] * qArray[i][j];
                    }
                    temporarySum /= QR[k][k];
                    for (int i = k; i < rowDimension; i++) {
                        qArray[i][j] += temporarySum * QR[i][k];
                    }
                }
            }
        }
        return Q;
    }

    /**
     * Gets the upper triangular factor, \(R\).
     *
     * @return  The upper triangular factor, \(R\), as a new matrix.
     */
    public Matrix getR() {
        Matrix R = new Matrix(columnDimension, columnDimension);
        double[][] rArray = R.getArray();
        for (int i = 0; i < columnDimension; i++) {
            for (int j = i; j < columnDimension; j++) {
                if (i < j) {
                    rArray[i][j] = QR[i][j];
                } else if (i == j) {
                    rArray[i][j] = rDiagonal[i];
                }
            }
        }
        return R;
    }

    /**
     * Gets the boolean value indicating whether or not the matrix whose decomposition is being computed is full rank.
     *
     * @return  A boolean value indicating whether or not the matrix whose decomposition is being computed is full rank.
     */
    public boolean isFullRank() {
        return isFullRank;
    }
}
