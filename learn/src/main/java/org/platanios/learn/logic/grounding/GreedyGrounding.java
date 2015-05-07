package org.platanios.learn.logic.grounding;

import org.platanios.learn.logic.LogicManager;
import org.platanios.learn.logic.formula.Atom;
import org.platanios.learn.logic.formula.Formula;
import org.platanios.learn.logic.formula.Negation;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GreedyGrounding<T, R> extends Grounding<T, R> {
    public GreedyGrounding(LogicManager<T, R> logicManager) {
        super(logicManager);
    }

    @Override
    boolean pruneGroundingAndSetCurrentPredicateTruthValue(Formula<T> formula, List<R> disjunctionComponentsSoFar) {
        if (currentPredicateTruthValue == null)
            if (formula instanceof Negation) // This is the important thing that PSL is doing while considering only the body variables.
                return true;
            else
                disjunctionComponentsSoFar.add(logicManager.logic().falseValue());
        else if (formula instanceof Atom)
            disjunctionComponentsSoFar.add(currentPredicateTruthValue);
        else
            disjunctionComponentsSoFar.add(logicManager.logic().negation(currentPredicateTruthValue));
        currentPredicateTruthValue = logicManager.logic().disjunction(disjunctionComponentsSoFar);
        return logicManager.logic().isSatisfied(currentPredicateTruthValue);
    }
}
