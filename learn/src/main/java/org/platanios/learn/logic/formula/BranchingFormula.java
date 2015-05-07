package org.platanios.learn.logic.formula;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class BranchingFormula<T> extends Formula<T> {
    List<Formula<T>> formulas;

    BranchingFormula(List<Formula<T>> formulas) {
        this.formulas = formulas;
    }

    @Override
    public Set<Variable<T>> getVariables() {
        Set<Variable<T>> variables = new HashSet<>();
        for (Formula<T> formula : formulas)
            variables.addAll(formula.getVariables());
        return variables;
    }

    @Override
    public List<Variable<T>> getOrderedVariables() {
        List<Variable<T>> variables = new ArrayList<>();
        for (Formula<T> formula : formulas)
            variables.addAll(formula.getOrderedVariables());
        return variables;
    }

    public int getNumberOfComponents() {
        return formulas.size();
    }

    public Formula<T> getComponent(int index) {
        return formulas.get(index);
    }

    public abstract String operatorToString();

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        for (int formulaIndex = 0; formulaIndex < formulas.size(); formulaIndex++) {
            stringBuilder.append(formulas.get(formulaIndex).toString());
            if (formulaIndex < formulas.size() - 1)
                stringBuilder.append(" ").append(operatorToString()).append(" ");
        }
        return stringBuilder.toString();
    }
}
