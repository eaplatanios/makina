package org.platanios.learn.logic.grounding;

import org.platanios.learn.logic.LogicManager;
import org.platanios.learn.logic.formula.Atom;
import org.platanios.learn.logic.formula.Formula;

import java.util.HashSet;
import java.util.List;
import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ExhaustiveGrounding extends Grounding {
    public ExhaustiveGrounding(LogicManager logicManager) {
        super(logicManager);
    }

    @Override
    public void ground(List<Formula> formulas) {
        for (int currentFormulaIndex = 0; currentFormulaIndex < formulas.size(); currentFormulaIndex++) {
            ground(formulas.get(currentFormulaIndex));
            groundedFormulas.put(currentFormulaIndex, new HashSet<>(groundedFormula));
            System.out.println("Generated " + groundedFormula.size() + " groundings for rule " + currentFormulaIndex); // TODO: Use a logger for this part.
        }
    }

    @Override
    boolean pruneGroundingAndSetCurrentPredicateTruthValue(Formula formula,
                                                           Map<Long, Long> variableAssignments,
                                                           List<Double> disjunctionComponentsSoFar) {
        if (currentPredicateTruthValue == null)
            disjunctionComponentsSoFar.add(logicManager.logic().falseValue());
        else if (formula instanceof Atom)
            disjunctionComponentsSoFar.add(currentPredicateTruthValue);
        else
            disjunctionComponentsSoFar.add(logicManager.logic().negation(currentPredicateTruthValue));
        currentPredicateTruthValue = logicManager.logic().disjunction(disjunctionComponentsSoFar);
        return logicManager.logic().isSatisfied(currentPredicateTruthValue);
    }
}
