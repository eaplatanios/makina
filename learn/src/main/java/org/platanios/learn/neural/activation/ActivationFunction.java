package org.platanios.learn.neural.activation;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface ActivationFunction {
    Vector getValue(Vector point);
    Matrix getGradient(Vector point);
}
