package org.platanios.learn.logic.formula;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class BranchingFormula extends Formula {
    List<Formula> formulas;

    BranchingFormula(List<Formula> formulas) {
        this.formulas = formulas;
    }

    @Override
    public Set<Variable> getVariables() {
        Set<Variable> variables = new HashSet<>();
        for (Formula formula : formulas)
            variables.addAll(formula.getVariables());
        return variables;
    }

    @Override
    public List<Variable> getOrderedVariables() {
        List<Variable> variables = new ArrayList<>();
        for (Formula formula : formulas)
            variables.addAll(formula.getOrderedVariables());
        return variables;
    }

    public int getNumberOfComponents() {
        return formulas.size();
    }

    public Formula getComponent(int index) {
        return formulas.get(index);
    }

    public List<Formula> getComponents() {
        return formulas;
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
