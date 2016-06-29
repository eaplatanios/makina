package makina.learn.logic;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface Logic {
    double conjunction(List<Double> arguments);
    double disjunction(List<Double> arguments);
    double negation(double argument);
    double trueValue();
    double falseValue();
    boolean isSatisfied(double value);
}
