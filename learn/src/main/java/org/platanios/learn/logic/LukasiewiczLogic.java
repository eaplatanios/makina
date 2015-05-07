package org.platanios.learn.logic;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LukasiewiczLogic implements Logic<Double> {
    @Override
    public Double conjunction(List<Double> arguments) {
        double result = 0;
        for (double argument : arguments)
            result = Math.max(0, result + argument - 1);
        return result;
    }

    @Override
    public Double disjunction(List<Double> arguments) {
        double result = 0;
        for (double argument : arguments)
            result = Math.min(1, result + argument);
        return result;
    }

    @Override
    public Double negation(Double argument) {
        return 1 - argument;
    }

    @Override
    public Double trueValue() {
        return 1.0;
    }

    @Override
    public Double falseValue() {
        return 0.0;
    }

    @Override
    public boolean isSatisfied(Double value) {
        return value == 1.0;
    }
}
