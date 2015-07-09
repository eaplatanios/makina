package org.platanios.learn.logic.formula;

import org.platanios.learn.logic.LogicManager;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Negation extends Formula {
    private Formula formula;

    public Negation(Formula formula) {
        this.formula = formula;
    }

    public Formula getFormula() {
        return formula;
    }

    @Override
    public Set<Variable> getVariables() {
        return formula.getVariables();
    }

    @Override
    public List<Variable> getOrderedVariables() {
        return formula.getOrderedVariables();
    }

    @Override
    public Double evaluate(LogicManager logicManager, Map<Long, Long> variableAssignments) {
        return logicManager.logic().negation(formula.evaluate(logicManager, variableAssignments));
    }

    @Override
    public Formula toDisjunctiveNormalForm() {
        if (formula instanceof Atom)
            return this;
        else if (formula instanceof Negation)
            return ((Negation) formula).formula.toDisjunctiveNormalForm();
        else if (formula instanceof Conjunction) {
            int numberOfComponents = ((Conjunction) formula).getNumberOfComponents();
            List<Formula> components = new ArrayList<>(numberOfComponents);
            for (int componentIndex = 0; componentIndex < numberOfComponents; componentIndex++)
                components.add(new Negation(((Conjunction) formula).getComponent(componentIndex)));
            return new Disjunction(components).toDisjunctiveNormalForm();
        } else if (formula instanceof Disjunction) {
            int numberOfComponents = ((Disjunction) formula).getNumberOfComponents();
            List<Formula> components = new ArrayList<>(numberOfComponents);
            for (int componentIndex = 0; componentIndex < numberOfComponents; componentIndex++)
                components.add(new Negation(((Disjunction) formula).getComponent(componentIndex)));
            return new Conjunction(components).toDisjunctiveNormalForm();
        } else {
            throw new IllegalStateException("The provided formula type is not supported by this conversion algorithm.");
        }
    }

    @Override
    public String toString() {
        if (formula instanceof Atom)
            return "!" + formula.toString();
        else
            return "!(" + formula.toString() + ")";
    }
}
