package org.platanios.learn.optimization;

import org.platanios.learn.optimization.function.ProbabilisticSoftLogicFunction.ProbabilisticSoftLogicSumFunctionTerm;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class SubProblemSolvers {
    // Suppress default constructor for noninstantiability
    private SubProblemSolvers() {
        throw new AssertionError();
    }

    public static void solveProbabilisticSoftLogicSubProblem(
            ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblem subProblem
    ) {
        ProbabilisticSoftLogicSumFunctionTerm objectiveTerm =
                (ProbabilisticSoftLogicSumFunctionTerm) subProblem.objectiveTerm;
        if (objectiveTerm.getLinearFunction().getValue(subProblem.variables) > 0) {
            subProblem.variables.set(
                    0,
                    subProblem.variables.size() - 1,
                    new NewtonSolver.Builder(
                            new ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemObjectiveFunction(
                                    objectiveTerm.getSubProblemObjectiveFunction(),
                                    subProblem.consensusVariables,
                                    subProblem.multipliers,
                                    subProblem.augmentedLagrangianParameter
                            ),
                            subProblem.variables).build().solve()
            );
            if (objectiveTerm.getLinearFunction().getValue(subProblem.variables) < 0) {
                subProblem.variables.set(
                        0,
                        subProblem.variables.size() - 1,
                        objectiveTerm.getLinearFunction().projectToHyperplane(subProblem.consensusVariables)
                );
            }
        }
    }
}
