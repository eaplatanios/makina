package org.platanios.learn.neural.network;

import org.platanios.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
abstract class LossFunction {
    final Vector correctOutput;

    LossFunction(Vector correctOutput) {
        this.correctOutput = correctOutput;
    }

    int inputSize() {
        return correctOutput.size();
    }

    abstract double value(Vector networkOutput);
    abstract Vector gradient(Vector networkOutput);
}
