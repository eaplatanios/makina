package org.platanios.learn.optimization;

import org.junit.Test;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.optimization.constraint.LinearEqualityConstraint;
import org.platanios.learn.optimization.function.ProbabilisticSoftLogicFunction;


/**
 * @author Emmanouil Antonios Platanios
 */
public class ConsensusAlternatingDirectionsMethodOfMultipliersSolverTest {
    @Test
    public void testSimpleConsensusADMM1() {
        ProbabilisticSoftLogicFunction pslFunction = new ProbabilisticSoftLogicFunction.Builder(2)
                .addTerm(new int[]{0}, new double[]{-1}, 1, 1, 1)
                .addTerm(new int[]{0, 1}, new double[]{1, -1}, 0, 1, 1)
                .addTerm(new int[]{1}, new double[] {-1}, 0, 1, 1)
                .build();

        ConsensusAlternatingDirectionsMethodOfMultipliersSolver consensusAlternatingDirectionsMethodOfMultipliersSolver =
                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder(pslFunction,
                                                                                    Vectors.dense(new double[]{0.5,0.5}))
                        .subProblemSolver(SubProblemSolvers::solveProbabilisticSoftLogicSubProblem)
                        .checkForObjectiveConvergence(false)
                        .checkForGradientConvergence(false)
                        .loggingLevel(5)
                        .build();

        Vector result = consensusAlternatingDirectionsMethodOfMultipliersSolver.solve();
        System.out.println(result.get(0));
        System.out.println(result.get(1));
        System.out.println('\n');
    }

    @Test
    public void testSimpleConsensusADMM2() {
        ProbabilisticSoftLogicFunction pslFunction = new ProbabilisticSoftLogicFunction.Builder(2)
                .addTerm(new int[]{0}, new double[]{-1}, 1, 1, 1)
                .addTerm(new int[]{0, 1}, new double[]{1, -1}, 0, 1, 1)
                .addTerm(new int[]{1}, new double[] {-1}, 0, 1, 1)
                .addTerm(new int[]{0}, new double[]{1}, -1, 1, 1)
                .addTerm(new int[]{0, 1}, new double[]{-1, 1}, 0, 1, 1)
                .addTerm(new int[]{1}, new double[] {1}, 0, 1, 1)
                .build();

        ConsensusAlternatingDirectionsMethodOfMultipliersSolver consensusAlternatingDirectionsMethodOfMultipliersSolver =
                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder(pslFunction,
                                                                                    Vectors.dense(new double[]{0.5,0.5}))
                        .subProblemSolver(SubProblemSolvers::solveProbabilisticSoftLogicSubProblem)
                        .checkForObjectiveConvergence(false)
                        .checkForGradientConvergence(false)
                        .loggingLevel(5)
                        .build();

        Vector result = consensusAlternatingDirectionsMethodOfMultipliersSolver.solve();
        System.out.println(result.get(0));
        System.out.println(result.get(1));
        System.out.println('\n');
    }

    @Test
    public void testSimpleConstrainedConsensusADMM2() {
        ProbabilisticSoftLogicFunction pslFunction = new ProbabilisticSoftLogicFunction.Builder(2)
                .addTerm(new int[]{0}, new double[]{-1}, 1, 1, 1)
                .addTerm(new int[]{0, 1}, new double[]{1, -1}, 0, 1, 1)
                .addTerm(new int[]{1}, new double[] {-1}, 0, 1, 1)
                .addTerm(new int[]{0}, new double[]{1}, -1, 1, 1)
                .addTerm(new int[]{0, 1}, new double[]{-1, 1}, 0, 1, 1)
                .addTerm(new int[]{1}, new double[] {1}, 0, 1, 1)
                .build();

        ConsensusAlternatingDirectionsMethodOfMultipliersSolver consensusAlternatingDirectionsMethodOfMultipliersSolver =
                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder(pslFunction,
                                                                                    Vectors.dense(new double[]{0.5,0.5}))
                        .addConstraint(new LinearEqualityConstraint(Vectors.dense(new double[]{-1, 1}), 0.2), 0, 1)
                        .subProblemSolver(SubProblemSolvers::solveProbabilisticSoftLogicSubProblem)
                        .checkForObjectiveConvergence(false)
                        .checkForGradientConvergence(false)
                        .loggingLevel(5)
                        .build();

        Vector result = consensusAlternatingDirectionsMethodOfMultipliersSolver.solve();
        System.out.println(result.get(0));
        System.out.println(result.get(1));
        System.out.println('\n');
    }
}
