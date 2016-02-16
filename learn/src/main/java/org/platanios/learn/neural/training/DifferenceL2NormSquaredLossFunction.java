package org.platanios.learn.neural.training;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorNorm;
import org.platanios.learn.neural.network.Network;
import org.platanios.learn.neural.network.State;
import org.platanios.learn.neural.network.Variable;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DifferenceL2NormSquaredLossFunction extends LossFunction {
    public DifferenceL2NormSquaredLossFunction(Network network) {
        super(network);
    }

    @Override
    public double value(State state, Vector correctOutput) {
        return network.value(state).sub(correctOutput).norm(VectorNorm.L2_SQUARED);
    }

    @Override
    public Vector gradient(State state, Variable variable, Vector correctOutput) {
        return network.value(state).sub(correctOutput).mult(2).transMult(network.gradient(state, variable));
    }
}
