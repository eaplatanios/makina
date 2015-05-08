package org.platanios.learn.logic.grounding;

import org.platanios.learn.logic.LogicManager;
import org.platanios.learn.logic.formula.Atom;
import org.platanios.learn.logic.formula.Formula;
import org.platanios.learn.logic.formula.Variable;

import java.util.List;
import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ExhaustiveGrounding<T, R> extends Grounding<T, R> {
    public ExhaustiveGrounding(LogicManager<T, R> logicManager) {
        super(logicManager);
    }

    @Override
    boolean pruneGroundingAndSetCurrentPredicateTruthValue(Formula<T> formula,
                                                           Map<Variable<T>, T> variableAssignments,
                                                           List<R> disjunctionComponentsSoFar) {
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
