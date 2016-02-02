package org.platanios.learn.classification.active;

import com.google.common.base.Objects;

/**
 * @author Emmanouil Antonios Platanios
 */
public class EntropyScoringFunction extends ScoringFunction {
    private final boolean propagateConstraints;
    private final boolean onlyConsiderLeafNodes;

    public EntropyScoringFunction() {
        this(false, false);
    }

    public EntropyScoringFunction(boolean propagateConstraints, boolean onlyConsiderLeafNodes) {
        this.propagateConstraints = propagateConstraints;
        this.onlyConsiderLeafNodes = onlyConsiderLeafNodes;
    }

    @Override
    public Double computeInformationGainHeuristicValue(Learning learning, Learning.InstanceToLabel instanceToLabel) {
        return entropy(instanceToLabel.getProbability());
    }

    private double entropy(double probability) {
        if (probability > 0 && probability < 1)
            return -probability * Math.log(probability) - (1 - probability) * Math.log(1 - probability);
        else
            return 0;
    }

    @Override
    public boolean propagateConstraints() {
        return propagateConstraints;
    }

    @Override
    public boolean onlyConsiderLeafNodes() {
        return onlyConsiderLeafNodes;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        EntropyScoringFunction that = (EntropyScoringFunction) other;

        return Objects.equal(propagateConstraints, that.propagateConstraints)
                && Objects.equal(onlyConsiderLeafNodes, that.onlyConsiderLeafNodes);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(this.getClass(), propagateConstraints, onlyConsiderLeafNodes);
    }

    @Override
    public String toString() {
        if (propagateConstraints && onlyConsiderLeafNodes)
            return "ENTROPY-LEAF-CP";
        else if (propagateConstraints)
            return "ENTROPY-CP";
        else if (onlyConsiderLeafNodes)
            return "ENTROPY-LEAF";
        else
            return "ENTROPY";
    }
}
