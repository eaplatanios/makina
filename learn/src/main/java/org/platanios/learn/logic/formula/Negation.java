package org.platanios.learn.logic.formula;

import org.platanios.learn.logic.LogicManager;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Negation<T> extends Formula<T> {
    private Formula<T> formula;

    public Negation(Formula<T> formula) {
        this.formula = formula;
    }

    public Formula<T> getFormula() {
        return formula;
    }

    @Override
    public Set<Variable<T>> getVariables() {
        return formula.getVariables();
    }

    @Override
    public List<Variable<T>> getOrderedVariables() {
        return formula.getOrderedVariables();
    }

    @Override
    public <R> R evaluate(LogicManager<T, R> logicManager, Map<Variable<T>, T> variableAssignments) {
        return logicManager.logic().negation(formula.evaluate(logicManager, variableAssignments));
    }

    @Override
    public Formula<T> toDisjunctiveNormalForm() {
        if (formula instanceof Atom)
            return this;
        else if (formula instanceof Negation)
            return ((Negation<T>) formula).formula.toDisjunctiveNormalForm();
        else if (formula instanceof Conjunction) {
            int numberOfComponents = ((Conjunction<T>) formula).getNumberOfComponents();
            List<Formula<T>> components = new ArrayList<>(numberOfComponents);
            for (int componentIndex = 0; componentIndex < numberOfComponents; componentIndex++)
                components.add(new Negation<>(((Conjunction<T>) formula).getComponent(componentIndex)));
            return new Disjunction<>(components).toDisjunctiveNormalForm();
        } else if (formula instanceof Disjunction) {
            int numberOfComponents = ((Disjunction<T>) formula).getNumberOfComponents();
            List<Formula<T>> components = new ArrayList<>(numberOfComponents);
            for (int componentIndex = 0; componentIndex < numberOfComponents; componentIndex++)
                components.add(new Negation<>(((Disjunction<T>) formula).getComponent(componentIndex)));
            return new Conjunction<>(components).toDisjunctiveNormalForm();
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
