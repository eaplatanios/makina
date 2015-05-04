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
 * A linear least squares function of the form \(f(x)=\frac{1}{2}\|Jx-y\|^2\).
 *
 * @author Emmanouil Antonios Platanios
 */
public final class LinearLeastSquaresFunction extends AbstractLeastSquaresFunction {
    private final Matrix J;
    private final Vector y;

    public LinearLeastSquaresFunction(Matrix J, Vector y) {
        this.J = J;
        this.y = y;
    }

    public Vector computeResiduals(Vector point) {
        return J.multiply(point).sub(y);
    }

    public Matrix computeJacobian(Vector point) {
        return J;
    }

    public Matrix getJ() {
        return J;
    }

    public Vector getY() {
        return y;
    }

    @Override
    public void write(OutputStream outputStream, boolean includeType) throws IOException {
        if (includeType) {
            UnsafeSerializationUtilities.writeInt(outputStream, FunctionType.LinearLeastSquaresFunction.ordinal());
        }
        this.J.write(outputStream);
        this.y.write(outputStream, true);
    }

    public static LinearLeastSquaresFunction read(InputStream inputStream, boolean includeType) throws IOException {
        if (includeType) {
            FunctionType functionType = FunctionType.values()[UnsafeSerializationUtilities.readInt(inputStream)];
            if (functionType != FunctionType.LinearLeastSquaresFunction) {
                throw new InvalidObjectException("The stored function is of type " + functionType.name() + "!");
            }
        }
        Matrix J = new Matrix(inputStream);
        Vector y = Vectors.build(inputStream);
        return new LinearLeastSquaresFunction(J, y);
    }
}
