package org.platanios.learn.logic.formula;

import org.platanios.learn.logic.LogicManager;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class Formula<T> {
    public class Builder {
        private Formula<T> currentFormula;

        public Builder(Formula<T> initialFormula) {
            currentFormula = initialFormula;
        }

//        public Builder conjunction(Formula<T> formula) {
//            currentFormula = new Conjunction<>(currentFormula, formula);
//            return this;
//        }
//
//        public Builder disjunction(Formula<T> formula) {
//            currentFormula = new Disjunction<>(currentFormula, formula);
//            return this;
//        }
//
//        public Builder negate() {
//            currentFormula = new Negation<>(currentFormula);
//            return this;
//        }
    }

    public abstract Set<Variable<T>> getVariables();
    public abstract List<Variable<T>> getOrderedVariables();
    public abstract <R> R evaluate(LogicManager<T, R> logicManager, Map<Variable<T>, T> variableAssignments);
    public abstract Formula<T> toDisjunctiveNormalForm();
    @Override
    public abstract String toString();
}
