package makina.learn.logic.formula;

import makina.learn.logic.LogicManager;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class Formula {
    public class Builder {
//        private Formula currentFormula;
//
//        public Builder(Formula initialFormula) {
//            currentFormula = initialFormula;
//        }
//
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

    public abstract Set<Variable> getVariables();
    public abstract List<Variable> getOrderedVariables();
    public abstract Double evaluate(LogicManager logicManager, Map<Long, Long> variableAssignments);
    public abstract Formula toDisjunctiveNormalForm();
    @Override
    public abstract String toString();
}
