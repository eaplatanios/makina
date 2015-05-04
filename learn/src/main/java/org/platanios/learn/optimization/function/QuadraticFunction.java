package org.platanios.learn.optimization.function;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.serialization.UnsafeSerializationUtilities;

import java.io.IOException;
import java.io.InputStream;
import java.io.InvalidObjectException;
import java.io.OutputStream;

/**
 * A quadratic function of the form \(f(x)=\frac{1}{2}x^TAx-b^Tx\).
 *
 * @author Emmanouil Antonios Platanios
 */
public final class QuadraticFunction extends AbstractFunction {
    private final Matrix A;
    private final Vector b;

    public QuadraticFunction(Matrix A, Vector b) {
        this.A = A;
        this.b = b;
    }

    @Override
    public double computeValue(Vector point) {
        return 0.5 * point.transMult(A).inner(point) - b.inner(point);
    }

    @Override
    public Vector computeGradient(Vector point) {
        return A.multiply(point).sub(b);
    }

    @Override
    public Matrix computeHessian(Vector point) {
        return A;
    }

    public Matrix getA() {
        return A;
    }

    public Vector getB() {
        return b;
    }

    @Override
    public void write(OutputStream outputStream, boolean includeType) throws IOException {
        if (includeType) {
            UnsafeSerializationUtilities.writeInt(outputStream, FunctionType.QuadraticFunction.ordinal());
        }
        this.A.write(outputStream);
        this.b.write(outputStream, true);
    }

    public static QuadraticFunction read(InputStream inputStream, boolean includeType) throws IOException {
        if (includeType) {
            FunctionType functionType = FunctionType.values()[UnsafeSerializationUtilities.readInt(inputStream)];
            if (functionType != FunctionType.QuadraticFunction) {
                throw new InvalidObjectException("The stored function is of type " + functionType.name() + "!");
            }
        }
        Matrix A = new Matrix(inputStream);
        Vector b = Vectors.build(inputStream);
        return new QuadraticFunction(A, b);
    }
}
