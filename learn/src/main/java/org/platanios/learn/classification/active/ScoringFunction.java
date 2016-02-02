package org.platanios.learn.classification.active;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class ScoringFunction {
    public abstract Double computeInformationGainHeuristicValue(Learning learning,
                                                                Learning.InstanceToLabel instanceToLabel);

    public boolean propagateConstraints() {
        return true;
    }

    public boolean onlyConsiderLeafNodes() {
        return false;
    }

    @Override
    public abstract boolean equals(Object other);

    @Override
    public abstract int hashCode();

    @Override
    public abstract String toString();
}
