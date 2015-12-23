package org.platanios.learn.classification.active;

import com.google.common.base.Objects;

import java.util.Random;

/**
 * @author Emmanouil Antonios Platanios
 */
public class RandomScoringFunction extends ScoringFunction {
    private final Random random;

    public RandomScoringFunction() {
        random = new Random();
    }

    public RandomScoringFunction(long randomSeed) {
        random = new Random(randomSeed);
    }

    @Override
    public Double computeInformationGainHeuristicValue(Learning learning, Learning.InstanceToLabel instanceToLabel) {
        return random.nextDouble();
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null || getClass() != other.getClass())
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(this.getClass());
    }

    @Override
    public String toString() {
        return "RANDOM";
    }
}
