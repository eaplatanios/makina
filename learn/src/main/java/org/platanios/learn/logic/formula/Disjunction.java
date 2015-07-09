package org.platanios.learn.logic.formula;

import org.platanios.learn.logic.LogicManager;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Disjunction extends BranchingFormula {
    public Disjunction(List<Formula> formulas) {
        super(formulas);
    }

    @Override
    public Double evaluate(LogicManager logicManager, Map<Long, Long> variableAssignments) {
        return logicManager.logic().disjunction(
                formulas.stream()
                        .map(formula -> formula.evaluate(logicManager, variableAssignments))
                        .collect(Collectors.toList())
        );
    }

    /**
     * Note that the resulting formula is flattened.
     *
     * @return
     */
    @Override
    public Formula toDisjunctiveNormalForm() {
        return new Disjunction(formulas.stream()
                                       .map(Formula::toDisjunctiveNormalForm)
                                       .collect(Collectors.toList())).flatten();
    }

    public Disjunction flatten() {
        ArrayList<Formula> disjunctionComponents = new ArrayList<>(formulas.size());
        for (Formula formula : formulas) {
            if (formula instanceof Disjunction)
                disjunctionComponents.addAll(((Disjunction) formula).flatten().formulas);
            else
                disjunctionComponents.add(formula);
        }
        return new Disjunction(disjunctionComponents);
    }

    @Override
    public String operatorToString() {
        return "|";
    }
}
