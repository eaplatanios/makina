package org.platanios.learn.logic;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LukasiewiczLogic implements Logic {
    @Override
    public double conjunction(List<Double> arguments) {
        double result = 0;
        for (double argument : arguments)
            result = Math.max(0, result + argument - 1);
        return result;
    }

    @Override
    public double disjunction(List<Double> arguments) {
        double result = 0;
        for (double argument : arguments)
            result = Math.min(1, result + argument);
        return result;
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
