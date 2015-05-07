package org.platanios.learn.logic.formula;

import org.platanios.learn.logic.LogicManager;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Disjunction<T> extends BranchingFormula<T> {
    public Disjunction(List<Formula<T>> formulas) {
        super(formulas);
    }

    @Override
    public <R> R evaluate(LogicManager<T, R> logicManager, Map<Variable<T>, T> variableAssignments) {
        return logicManager.logic().disjunction(formulas.stream()
                                                        .map(formula -> formula.evaluate(logicManager, variableAssignments))
                                                        .collect(Collectors.toList()));
    }

    /**
     * Note that the resulting formula is flattened.
     *
     * @return
     */
    @Override
    public Formula<T> toDisjunctiveNormalForm() {
        return new Disjunction<>(formulas.stream()
                                       .map(Formula::toDisjunctiveNormalForm)
                                       .collect(Collectors.toList())).flatten();
    }

    public Disjunction<T> flatten() {
        ArrayList<Formula<T>> disjunctionComponents = new ArrayList<>(formulas.size());
        for (Formula<T> formula : formulas) {
            if (formula instanceof Disjunction)
                disjunctionComponents.addAll(((Disjunction<T>) formula).flatten().formulas);
            else
                disjunctionComponents.add(formula);
        }
        return new Disjunction<>(disjunctionComponents);
    }

    @Override
    public String operatorToString() {
        return "|";
    }
}
