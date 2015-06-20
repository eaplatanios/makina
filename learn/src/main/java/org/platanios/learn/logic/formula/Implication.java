package org.platanios.learn.logic.formula;

import org.platanios.learn.logic.LogicManager;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Implication extends Formula {
    private Formula bodyFormula;
    private Formula headFormula;

    public Implication(Formula bodyFormula, Formula headFormula) {
        this.bodyFormula = bodyFormula;
        this.headFormula = headFormula;
    }

    @Override
    public Set<Variable> getVariables() {
        Set<Variable> variables = bodyFormula.getVariables();
        variables.addAll(headFormula.getVariables());
        return variables;
    }

    @Override
    public List<Variable> getOrderedVariables() {
        List<Variable> variables = bodyFormula.getOrderedVariables();
        variables.addAll(headFormula.getOrderedVariables());
        return variables;
    }

    @Override
    public <R> R evaluate(LogicManager<R> logicManager, Map<Long, Long> variableAssignments) {
        List<R> components = new ArrayList<>(2);
        components.add(logicManager.logic().negation(bodyFormula.evaluate(logicManager, variableAssignments)));
        components.add(headFormula.evaluate(logicManager, variableAssignments));
        return logicManager.logic().disjunction(components);
    }

    @Override
    public Formula toDisjunctiveNormalForm() {
        List<Formula> disjunctionComponents = new ArrayList<>(2);
        disjunctionComponents.add(new Negation(bodyFormula).toDisjunctiveNormalForm());
        disjunctionComponents.add(headFormula.toDisjunctiveNormalForm());
        return new Disjunction(disjunctionComponents);
    }

    @Override
    public String toString() {
        return bodyFormula.toString() + " -> " + headFormula.toString();
    }
}
