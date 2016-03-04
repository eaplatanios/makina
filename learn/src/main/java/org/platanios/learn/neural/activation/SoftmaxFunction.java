package org.platanios.learn.neural.activation;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SoftmaxFunction {
    public static Vector value(Vector point) {
        double expSum = point.map(Math::exp).sum();
        Vector value = Vectors.build(point.size(), point.type());
        for (Vector.VectorElement element : point)
            value.set(element.index(), Math.exp(element.value()) / expSum);
        return value;
    }

    public static Matrix gradient(Vector point) {
        Vector value = value(point);
        Matrix gradient = new Matrix(point.size(), point.size());
        for (int outputIndex = 0; outputIndex < point.size(); outputIndex++)
            for (int inputIndex = 0; inputIndex < point.size(); inputIndex++) {
                gradient.setElement(
                        outputIndex,
                        inputIndex,
                        value.get(outputIndex) * ((outputIndex == inputIndex ? 1.0 : 0.0) - value.get(inputIndex))
                );
            }
        return gradient;
    }
}
