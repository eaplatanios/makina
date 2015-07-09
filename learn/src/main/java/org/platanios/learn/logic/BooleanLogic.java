package org.platanios.learn.logic;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class BooleanLogic implements Logic {
    @Override
    public double conjunction(List<Double> arguments) {
        for (double argument : arguments)
            if (argument == 0)
                return 0;
        return 1;
    }

    @Override
    public double disjunction(List<Double> arguments) {
        for (double argument : arguments)
            if (argument == 1)
                return 1;
        return 0;
    }

    @Override
    public double negation(double argument) {
        return 1 - argument;
    }

    @Override
    public double trueValue() {
        return 1;
    }

    @Override
    public double falseValue() {
        return 0;
    }

    @Override
    public boolean isSatisfied(double value) {
        return value == 1;
    }
}
