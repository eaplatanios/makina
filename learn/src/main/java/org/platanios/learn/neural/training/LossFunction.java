package org.platanios.learn.neural.training;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.neural.network.Network;
import org.platanios.learn.neural.network.State;
import org.platanios.learn.neural.network.Variable;

import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class LossFunction {
    protected final Network network;

    public LossFunction(Network network) {
        this.network = network;
    }

    public abstract double value(State state, Vector correctOutput);
    public abstract Vector gradient(State state, Variable variable, Vector correctOutput);

    public List<Vector> gradient(State state, List<Variable> variables, Vector correctOutput) {
        return variables.stream()
                .map(variable -> gradient(state, variable, correctOutput))
                .collect(Collectors.toList());
    }
}
