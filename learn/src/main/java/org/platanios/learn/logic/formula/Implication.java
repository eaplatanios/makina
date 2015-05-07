package org.platanios.learn.logic.formula;

import org.platanios.learn.logic.LogicManager;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Implication<T> extends Formula<T> {
    private Formula<T> bodyFormula;
    private Formula<T> headFormula;

    public Implication(Formula<T> bodyFormula, Formula<T> headFormula) {
        this.bodyFormula = bodyFormula;
        this.headFormula = headFormula;
    }

    @Override
    public Set<Variable<T>> getVariables() {
        Set<Variable<T>> variables = bodyFormula.getVariables();
        variables.addAll(headFormula.getVariables());
        return variables;
    }

    @Override
    public List<Variable<T>> getOrderedVariables() {
        List<Variable<T>> variables = bodyFormula.getOrderedVariables();
        variables.addAll(headFormula.getOrderedVariables());
        return variables;
    }

    @Override
    public <R> R evaluate(LogicManager<T, R> logicManager, Map<Variable<T>, T> variableAssignments) {
        List<R> components = new ArrayList<>(2);
        components.add(logicManager.logic().negation(bodyFormula.evaluate(logicManager, variableAssignments)));
        components.add(headFormula.evaluate(logicManager, variableAssignments));
        return logicManager.logic().disjunction(components);
    }

    @Override
    public Formula<T> toDisjunctiveNormalForm() {
        List<Formula<T>> disjunctionComponents = new ArrayList<>(2);
        disjunctionComponents.add(new Negation<>(bodyFormula).toDisjunctiveNormalForm());
        disjunctionComponents.add(headFormula.toDisjunctiveNormalForm());
        return new Disjunction<>(disjunctionComponents);
    }

    @Override
    public String toString() {
        return bodyFormula.toString() + " -> " + headFormula.toString();
    }
}
