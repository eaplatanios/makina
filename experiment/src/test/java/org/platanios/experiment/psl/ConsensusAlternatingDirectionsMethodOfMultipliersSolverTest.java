package org.platanios.experiment.psl;

import org.junit.Test;

import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ConsensusAlternatingDirectionsMethodOfMultipliersSolverTest {
    @Test
    public void testSimpleConsensusADMM1() {
        ProbabilisticSoftLogicProblem pslProblem = new ProbabilisticSoftLogicProblem.Builder(new int[]{1, 3},
                                                                                             new double[]{1, 0},
                                                                                             2)
                .addRule(new int[]{0}, new int[]{1}, new boolean[]{false}, new boolean[]{false}, 1, 1)
                .addRule(new int[]{2}, new int[]{0}, new boolean[]{false}, new boolean[]{false}, 1, 1)
                .addRule(new int[]{2}, new int[]{3}, new boolean[]{false}, new boolean[]{false}, 1, 1)
                .build();
        Map<Integer, Double> result = pslProblem.solve();
        System.out.println(result.get(0));
        System.out.println(result.get(2));
        System.out.println('\n');
    }

//    @Test
//    public void testSimpleConsensusADMM1_NewInterface() {
//        ProbabilisticSoftLogicFunction pslFunction = new ProbabilisticSoftLogicFunction.Builder(2)
//                .addRule(new int[]{0}, new int[]{2}, new boolean[]{false}, new boolean[]{false}, new int[]{2}, new double[]{1}, 1, 1)           // C -> A  =>  1 - A
//                .addRule(new int[]{1}, new int[]{0}, new boolean[]{false}, new boolean[]{false}, new int[]{}, new double[]{}, 1, 1)             // A -> B  =>  A - B
//                .addRule(new int[]{1}, new int[]{3}, new boolean[]{false}, new boolean[]{false}, new int[]{3}, new double[]{0}, 1, 1)           // D -> B  =>  -B
//                .build();
//
//        ConsensusAlternatingDirectionsMethodOfMultipliersSolver consensusAlternatingDirectionsMethodOfMultipliersSolver =
//                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder(pslFunction,
//                                                                                    Vectors.dense(new double[]{0.5,0.5}))
//                        .subProblemSolver(SubProblemSolvers::solveProbabilisticSoftLogicSubProblem)
//                        .checkForObjectiveConvergence(false)
//                        .checkForGradientConvergence(false)
//                        .loggingLevel(5)
//                        .build();
//
//        Vector result = consensusAlternatingDirectionsMethodOfMultipliersSolver.solve();
//        System.out.println(result.get(0));
//        System.out.println(result.get(1));
//        System.out.println('\n');
//    }
//
//    @Test
//    public void testSimpleConsensusADMM2() {
//        ProbabilisticSoftLogicFunction pslFunction = new ProbabilisticSoftLogicFunction.Builder(2)
//                .addTerm(new int[]{0}, new double[]{-1}, 1, 1, 1)
//                .addTerm(new int[]{0, 1}, new double[]{1, -1}, 0, 1, 1)
//                .addTerm(new int[]{1}, new double[] {-1}, 0, 1, 1)
//                .addTerm(new int[]{0}, new double[]{1}, -1, 1, 1)
//                .addTerm(new int[]{0, 1}, new double[]{-1, 1}, 0, 1, 1)
//                .addTerm(new int[]{1}, new double[] {1}, 0, 1, 1)
//                .build();
//
//        ConsensusAlternatingDirectionsMethodOfMultipliersSolver consensusAlternatingDirectionsMethodOfMultipliersSolver =
//                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder(pslFunction,
//                                                                                    Vectors.dense(new double[]{0.5,0.5}))
//                        .subProblemSolver(SubProblemSolvers::solveProbabilisticSoftLogicSubProblem)
//                        .checkForObjectiveConvergence(false)
//                        .checkForGradientConvergence(false)
//                        .loggingLevel(5)
//                        .build();
//
//        Vector result = consensusAlternatingDirectionsMethodOfMultipliersSolver.solve();
//        System.out.println(result.get(0));
//        System.out.println(result.get(1));
//        System.out.println('\n');
//    }
//
//    @Test
//    public void testSimpleConstrainedConsensusADMM2() {
//        ProbabilisticSoftLogicFunction pslFunction = new ProbabilisticSoftLogicFunction.Builder(2)
//                .addTerm(new int[]{0}, new double[]{-1}, 1, 1, 1)
//                .addTerm(new int[]{0, 1}, new double[]{1, -1}, 0, 1, 1)
//                .addTerm(new int[]{1}, new double[] {-1}, 0, 1, 1)
//                .addTerm(new int[]{0}, new double[]{1}, -1, 1, 1)
//                .addTerm(new int[]{0, 1}, new double[]{-1, 1}, 0, 1, 1)
//                .addTerm(new int[]{1}, new double[] {1}, 0, 1, 1)
//                .build();
//
//        ConsensusAlternatingDirectionsMethodOfMultipliersSolver consensusAlternatingDirectionsMethodOfMultipliersSolver =
//                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder(pslFunction,
//                                                                                    Vectors.dense(new double[]{0.5,0.5}))
//                        .addConstraint(new LinearEqualityConstraint(Vectors.dense(new double[]{-1, 1}), 0.2), 0, 1)
//                        .subProblemSolver(SubProblemSolvers::solveProbabilisticSoftLogicSubProblem)
//                        .checkForObjectiveConvergence(false)
//                        .checkForGradientConvergence(false)
//                        .loggingLevel(5)
//                        .build();
//
//        Vector result = consensusAlternatingDirectionsMethodOfMultipliersSolver.solve();
//        System.out.println(result.get(0));
//        System.out.println(result.get(1));
//        System.out.println('\n');
//    }
//
//
//    @Test
//    public void testSimpleConsensusSquaredADMM() {
//        ProbabilisticSoftLogicFunction pslFunction = new ProbabilisticSoftLogicFunction.Builder(2)
//                .addTerm(new int[]{0}, new double[]{-1}, 1, 2, 1)
//                .addTerm(new int[]{0, 1}, new double[]{1, -1}, 0, 2, 1)
//                .addTerm(new int[]{1}, new double[] {-1}, 0, 2, 1)
//                .addTerm(new int[]{0}, new double[]{1}, -1, 2, 1)
//                .addTerm(new int[]{0, 1}, new double[]{-1, 1}, 0, 2, 1)
//                .addTerm(new int[]{1}, new double[]{1}, 0, 2, 1)
//                .build();
//
//        ConsensusAlternatingDirectionsMethodOfMultipliersSolver consensusAlternatingDirectionsMethodOfMultipliersSolver =
//                new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.Builder(pslFunction,
//                        Vectors.dense(new double[]{0.5,0.5}))
//                        .subProblemSolver(SubProblemSolvers::solveProbabilisticSoftLogicSubProblem)
//                        .checkForObjectiveConvergence(false)
//                        .checkForGradientConvergence(false)
//                        .loggingLevel(5)
//                        .build();
//
//        Vector result = consensusAlternatingDirectionsMethodOfMultipliersSolver.solve();
//        System.out.println(result.get(0));
//        System.out.println(result.get(1));
//        System.out.println('\n');
//    }
}
