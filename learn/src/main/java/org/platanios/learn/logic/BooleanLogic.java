package org.platanios.learn.logic;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class BooleanLogic implements Logic<Boolean> {
    @Override
    public Boolean conjunction(List<Boolean> arguments) {
        Boolean result = true;
        for (Boolean argument : arguments)
            result &= argument;
        return result;
    }

    @Override
    public Boolean disjunction(List<Boolean> arguments) {
        Boolean result = false;
        for (Boolean argument : arguments)
            result |= argument;
        return result;
    }

    @Override
    public Boolean negation(Boolean argument) {
        return !argument;
    }

    @Override
    public Boolean trueValue() {
        return true;
    }

    @Override
    public Boolean falseValue() {
        return false;
    }

    @Override
    public boolean isSatisfied(Boolean value) {
        return value;
    }
}
