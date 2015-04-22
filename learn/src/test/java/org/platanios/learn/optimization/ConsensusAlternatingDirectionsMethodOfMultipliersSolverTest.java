package org.platanios.learn.optimization;

import org.junit.Test;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.optimization.constraint.LinearEqualityConstraint;
import org.platanios.learn.optimization.function.LinearFunction;
import org.platanios.learn.optimization.function.SumFunction;


/**
 * Created by dcard on 4/21/15.
 */
public class ConsensusAlternatingDirectionsMethodOfMultipliersSolverTest {
    @Test
    public void testSimpleConsensusADMM1() {
        SumFunction sumFunction = new SumFunction.Builder(2)
                .addTerm(new LinearFunction(Vectors.dense(new double[]{-1}), 1), 0)
                .addTerm(new LinearFunction(Vectors.dense(new double[]{1, -1}), 0), 0, 1)
                .addTerm(new LinearFunction(Vectors.dense(new double[]{-1}), 0), 1)
                .build();

        ConsensusAlternatingDirectionsMethodOfMultipliersSolver consensusAlternatingDirectionsMethodOfMultipliersSolver =
                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder( sumFunction,
                        Vectors.dense(new double[]{0.5,0.5}))
                        .maximumNumberOfIterations(100)
                        .checkForObjectiveConvergence(false)
                        .checkForGradientConvergence(false)
                        .checkForPointConvergence(false)
                        .loggingLevel(5)
                        .build();

        Vector result = consensusAlternatingDirectionsMethodOfMultipliersSolver.solve();
        System.out.println(result.get(0));
        System.out.println(result.get(1));
        System.out.println('\n');
    }

    @Test
    public void testSimpleConsensusADMM2() {
        SumFunction sumFunction = new SumFunction.Builder(2)
                .addTerm(new LinearFunction(Vectors.dense(new double[]{-1}), 1), 0)
                .addTerm(new LinearFunction(Vectors.dense(new double[]{1, -1}), 0), 0, 1)
                .addTerm(new LinearFunction(Vectors.dense(new double[]{-1}), 0), 1)
                .addTerm(new LinearFunction(Vectors.dense(new double[]{1}), -1), 0)
                .addTerm(new LinearFunction(Vectors.dense(new double[]{-1, 1}), 0), 0, 1)
                .addTerm(new LinearFunction(Vectors.dense(new double[]{1}), 0), 1)
                .build();

        ConsensusAlternatingDirectionsMethodOfMultipliersSolver consensusAlternatingDirectionsMethodOfMultipliersSolver =
                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder( sumFunction,
                        Vectors.dense(new double[]{0.5,0.5}))
                        .maximumNumberOfIterations(100)
                        .checkForObjectiveConvergence(false)
                        .checkForGradientConvergence(false)
                        .checkForPointConvergence(false)
                        .loggingLevel(5)
                        .build();

        Vector result = consensusAlternatingDirectionsMethodOfMultipliersSolver.solve();
        System.out.println(result.get(0));
        System.out.println(result.get(1));
        System.out.println('\n');
    }

    @Test
    public void testSimpleConstrainedConsensusADMM2() {
        SumFunction sumFunction = new SumFunction.Builder(2)
                .addTerm(new LinearFunction(Vectors.dense(new double[]{-1}), 1), 0)
                .addTerm(new LinearFunction(Vectors.dense(new double[]{1, -1}), 0), 0, 1)
                .addTerm(new LinearFunction(Vectors.dense(new double[]{-1}), 0), 1)
                .addTerm(new LinearFunction(Vectors.dense(new double[]{1}), -1), 0)
                .addTerm(new LinearFunction(Vectors.dense(new double[]{-1, 1}), 0), 0, 1)
                .addTerm(new LinearFunction(Vectors.dense(new double[]{1}), 0), 1)
                .build();

        ConsensusAlternatingDirectionsMethodOfMultipliersSolver consensusAlternatingDirectionsMethodOfMultipliersSolver =
                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder(sumFunction, Vectors.dense(new double[]{0.5,0.5}))
                        .addConstraint(new int[]{0, 1}, new LinearEqualityConstraint(Vectors.dense(new double[]{-1, 1}), 0.2))
                        .augmentedLagrangianParameter(1)
                        .maximumNumberOfIterations(100)
                        .checkForObjectiveConvergence(false)
                        .checkForGradientConvergence(false)
                        .checkForPointConvergence(false)
                        .loggingLevel(5)
                        .build();

        Vector result = consensusAlternatingDirectionsMethodOfMultipliersSolver.solve();
        System.out.println(result.get(0));
        System.out.println(result.get(1));
        System.out.println('\n');
    }
}
