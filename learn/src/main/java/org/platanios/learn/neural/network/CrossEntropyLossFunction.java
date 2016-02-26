package org.platanios.learn.neural.network;

import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class CrossEntropyLossFunction extends LossFunction {
    public CrossEntropyLossFunction(Vector correctOutput) {
        super(correctOutput);
    }

    @Override
    double value(Vector networkOutput) {
        return correctOutput
                .multElementwise(networkOutput.map(Math::log))
                .multInPlace(-1)
                .subInPlace(correctOutput
                                    .map(x -> Math.log(1 - x))
                                    .multElementwise(networkOutput.map(x -> Math.log(1 - x))))
                .sum();
    }

    @Override
    Vector gradient(Vector networkOutput) {
        return correctOutput
                .map(x -> 1 - x)
                .divElementwise(networkOutput.map(x -> 1 - x))
                .subInPlace(correctOutput.divElementwise(networkOutput));
    }
}
