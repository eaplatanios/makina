package org.platanios.learn.logic;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface Logic<R> {
    R conjunction(List<R> arguments);
    R disjunction(List<R> arguments);
    R negation(R argument);
    R trueValue();
    R falseValue();
    boolean isSatisfied(R value);
}
