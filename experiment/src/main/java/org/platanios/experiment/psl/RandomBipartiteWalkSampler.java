package org.platanios.experiment.psl;

import com.google.common.collect.ImmutableList;
import org.apache.commons.lang3.ArrayUtils;
import org.platanios.learn.math.statistics.StatisticsUtilities;
import org.platanios.learn.optimization.ConsensusAlternatingDirectionsMethodOfMultipliersSolver;

import java.util.*;

/**
 * Class for
 */
public class RandomBipartiteWalkSampler implements ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemSelector {

    private final List<Integer> graphSampleOrigins;
    private final List<Integer> graphSampleCursorOriginIndices;
    private final List<Integer> graphSampleCursor;
    private final Random random;
    private final double restartProbability;
    private final double sampleProbability;
    private final Map<Integer, List<Integer>> predicateToTerms;
    private final RandomWalkSampler.TermPredicateIdGetter predicateIdGetter;
    private Map<Integer, Integer> internalToExternalIds;

    private RandomBipartiteWalkSampler(Builder builder) {

        this.random = builder.random;
        this.graphSampleOrigins = builder.originPredicates;
        this.graphSampleCursor = new ArrayList<>();
        this.graphSampleCursorOriginIndices = new ArrayList<>();
        this.restartProbability = builder.restartProbability;
        this.sampleProbability = builder.sampleProbability;
        this.predicateToTerms = builder.predicateToTerms;
        this.predicateIdGetter = builder.predicateIdGetter;
        this.internalToExternalIds = builder.internalToExternalIds;

    }

    public int[] selectSubProblems(ConsensusAlternatingDirectionsMethodOfMultipliersSolver solver) {

        // choose which seed to start each cursor from
        for (int iCursor = this.graphSampleCursorOriginIndices.size(); iCursor < solver.numberOfSubProblemSamples; ++iCursor) {
            int iSeed = random.nextInt(this.graphSampleOrigins.size());
            this.graphSampleCursorOriginIndices.add(iSeed);
            this.graphSampleCursor.add(-1);
        }

        HashSet<Integer> subProblemsToSample = new HashSet<>();

        for (int i = 0; i < solver.numberOfSubProblemSamples; ++i) {

            // first choose whether to restart
            double randRestart = this.random.nextDouble();
            if (randRestart < this.restartProbability) {
                this.graphSampleCursor.set(i, -1);
            } else {
                boolean isSampleOnStep = this.random.nextDouble() < this.sampleProbability;
                // first step, from predicate to term
                if (this.graphSampleCursor.get(i) == -1) {
                    this.graphSampleCursor.set(i, stepFromPredicate(this.graphSampleOrigins.get(this.graphSampleCursorOriginIndices.get(i))));
                } else { // other steps, from term to term
                    int predicate = stepFromTerm(this.graphSampleCursor.get(i));
                    this.graphSampleCursor.set(i, stepFromPredicate(predicate));
                }
                if (isSampleOnStep) {
                    subProblemsToSample.add(this.graphSampleCursor.get(i));
                }
            }
        }

        // back-fill with uniform
        while (subProblemsToSample.size() < solver.numberOfSubProblemSamples) {
            subProblemsToSample.add(this.random.nextInt(solver.getNumberOfTerms()));
        }

        int[] sampledSubProblems = new int[solver.numberOfSubProblemSamples];
        Integer[] sampledFromGraph = StatisticsUtilities.sampleWithoutReplacement(subProblemsToSample.toArray(new Integer[subProblemsToSample.size()]), Math.min(solver.numberOfSubProblemSamples, subProblemsToSample.size()));
        for(int i = 0; i < sampledFromGraph.length; ++i) {
            sampledSubProblems[i] = sampledFromGraph[i];
        }

        return sampledSubProblems;

    }

    private int stepFromPredicate(int predicate) {
        List<Integer> terms = this.predicateToTerms.get(predicate);
        return terms.get(this.random.nextInt(terms.size()));
    }

    private int stepFromTerm(int term) {
        int[] predicates = this.predicateIdGetter.getInternalPredicateIds(term);
        return this.internalToExternalIds.get(predicates[this.random.nextInt(predicates.length)]);
    }


    public static class Builder {

        private Random random;
        private List<Integer> originPredicates = new ArrayList<>();
        private final double restartProbability;
        private final double sampleProbability;
        private Map<Integer, Integer> internalToExternalIds;
        private Map<Integer, List<Integer>> predicateToTerms;
        private RandomWalkSampler.TermPredicateIdGetter predicateIdGetter;

        public Builder(
                Map<Integer, List<Integer>> predicateIdToTerms,
                Map<Integer, Integer> internalToExternalIds,
                RandomWalkSampler.TermPredicateIdGetter predicateIdGetter,
                double restartProbability,
                double sampleProbability,
                Random random) {

            this.random = random;

            this.restartProbability = restartProbability;
            this.sampleProbability = sampleProbability;

            this.internalToExternalIds = internalToExternalIds;
            this.predicateIdGetter = predicateIdGetter;
            this.predicateToTerms = predicateIdToTerms;

        }

        public Builder addOriginPredicate(int predicate) {
            if (this.predicateToTerms.containsKey(predicate)) {
                this.originPredicates.add(predicate);
            }
            return this;
        }

        public RandomBipartiteWalkSampler build() {
            return new RandomBipartiteWalkSampler(this);
        }

    }

}