package org.platanios.learn.neural.network;

import org.platanios.math.matrix.Vector;
import org.platanios.math.matrix.VectorNorm;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DifferenceL2NormSquaredLossFunction extends LossFunction {
    public DifferenceL2NormSquaredLossFunction(Vector correctOutput) {
        super(correctOutput);
    }

    @Override
    double value(Vector networkOutput) {
        return networkOutput.sub(correctOutput).norm(VectorNorm.L2_SQUARED);
    }

    @Override
    Vector gradient(Vector networkOutput) {
        return networkOutput.sub(correctOutput).mult(2);
    }
}
