package makina.learn.classification.active;

import com.google.common.base.Objects;

import java.util.Random;

/**
 * @author Emmanouil Antonios Platanios
 */
public class RandomScoringFunction extends ScoringFunction {
    private final Random random;
    private final boolean propagateConstraints;

    public RandomScoringFunction() {
        random = new Random();
        propagateConstraints = false;
    }

    public RandomScoringFunction(boolean propagateConstraints) {
        random = new Random();
        this.propagateConstraints = propagateConstraints;
    }

    public RandomScoringFunction(long randomSeed) {
        random = new Random(randomSeed);
        propagateConstraints = false;
    }

    public RandomScoringFunction(boolean propagateConstraints, boolean onlyConsiderLeafNodes, long randomSeed) {
        random = new Random(randomSeed);
        this.propagateConstraints = propagateConstraints;
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
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        RandomScoringFunction that = (RandomScoringFunction) other;

        return Objects.equal(propagateConstraints, that.propagateConstraints);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(this.getClass(), propagateConstraints);
    }

    @Override
    public String toString() {
        if (propagateConstraints)
            return "RANDOM-CP";
        else
            return "RANDOM";
    }
}
