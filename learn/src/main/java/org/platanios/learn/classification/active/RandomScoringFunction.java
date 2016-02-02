package org.platanios.learn.classification.active;

import com.google.common.base.Objects;

import java.util.Random;

/**
 * @author Emmanouil Antonios Platanios
 */
public class RandomScoringFunction extends ScoringFunction {
    private final Random random;
    private final boolean propagateConstraints;
    private final boolean onlyConsiderLeafNodes;

    public RandomScoringFunction() {
        random = new Random();
        propagateConstraints = false;
        onlyConsiderLeafNodes = false;
    }

    public RandomScoringFunction(boolean propagateConstraints, boolean onlyConsiderLeafNodes) {
        random = new Random();
        this.propagateConstraints = propagateConstraints;
        this.onlyConsiderLeafNodes = onlyConsiderLeafNodes;
    }

    public RandomScoringFunction(long randomSeed) {
        random = new Random(randomSeed);
        propagateConstraints = false;
        onlyConsiderLeafNodes = false;
    }

    public RandomScoringFunction(boolean propagateConstraints, boolean onlyConsiderLeafNodes, long randomSeed) {
        random = new Random(randomSeed);
        this.propagateConstraints = propagateConstraints;
        this.onlyConsiderLeafNodes = onlyConsiderLeafNodes;
    }

    @Override
    public Double computeInformationGainHeuristicValue(Learning learning, Learning.InstanceToLabel instanceToLabel) {
        return random.nextDouble();
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

        RandomScoringFunction that = (RandomScoringFunction) other;

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
            return "RANDOM-LEAF-CP";
        else if (propagateConstraints)
            return "RANDOM-CP";
        else if (onlyConsiderLeafNodes)
            return "RANDOM-LEAF";
        else
            return "RANDOM";
    }
}
