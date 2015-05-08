package org.platanios.learn.logic.grounding;

import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.platanios.learn.logic.LogicManager;
import org.platanios.learn.logic.formula.Atom;
import org.platanios.learn.logic.formula.Formula;
import org.platanios.learn.logic.formula.Negation;
import org.platanios.learn.logic.formula.Variable;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LazyGrounding<T, R> extends Grounding<T, R> {
    private Set<ActivatedGroundedPredicate<T>> activatedGroundedPredicates = new HashSet<>();

    public LazyGrounding(LogicManager<T, R> logicManager) {
        super(logicManager);
    }

    public LazyGrounding(LogicManager<T, R> logicManager,
                         Set<ActivatedGroundedPredicate<T>> activatedGroundedPredicates) {
        super(logicManager);
        this.activatedGroundedPredicates = activatedGroundedPredicates;
    }

    public Set<ActivatedGroundedPredicate<T>> getActivatedGroundedPredicates() {
        return activatedGroundedPredicates;
    }

    @Override
    boolean pruneGroundingAndSetCurrentPredicateTruthValue(Formula<T> formula,
                                                           Map<Variable<T>, T> variableAssignments,
                                                           List<R> disjunctionComponentsSoFar) {
        if (currentPredicateTruthValue == null) // This is the important thing that PSL is doing while considering only the body variables.
            if (formula instanceof Negation
                    && !activatedGroundedPredicates.contains(
                    new ActivatedGroundedPredicate<>(
                            ((Atom<T>) ((Negation<T>) formula).getFormula()).getPredicate().getIdentifier(),
                            ((Atom<T>) ((Negation<T>) formula).getFormula()).getPredicateArguments().stream().map(variableAssignments::get).collect(Collectors.toList())
                    ))) {
                return true;
            } else
                disjunctionComponentsSoFar.add(logicManager.logic().falseValue());
        else if (formula instanceof Atom)
            disjunctionComponentsSoFar.add(currentPredicateTruthValue);
        else
            disjunctionComponentsSoFar.add(logicManager.logic().negation(currentPredicateTruthValue));
        currentPredicateTruthValue = logicManager.logic().disjunction(disjunctionComponentsSoFar);
        return logicManager.logic().isSatisfied(currentPredicateTruthValue);
    }

    @Override
    void onGroundedPredicateAddition(List<GroundedPredicate<T, R>> groundedRule) {
        activatedGroundedPredicates.addAll(groundedRule.stream().filter(rule -> rule.getValue() == null).map(
                groundedPredicate ->
                        new ActivatedGroundedPredicate<>(groundedPredicate.getPredicate().getIdentifier(),
                                                         groundedPredicate.getPredicateArgumentsAssignment()))
                                                   .collect(Collectors.toList()));
    }

    public class ActivatedGroundedPredicate<V> {
        private final long predicateIdentifier;
        private final List<V> variableGroundings;

        private ActivatedGroundedPredicate(long predicateIdentifier, List<V> variableGroundings) {
            this.predicateIdentifier = predicateIdentifier;
            this.variableGroundings = variableGroundings;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        @SuppressWarnings("unchecked")
        public boolean equals(Object object) {
            if (!(object instanceof ActivatedGroundedPredicate))
                return false;
            if (object == this)
                return true;

            ActivatedGroundedPredicate<V> that = (ActivatedGroundedPredicate<V>) object;

            if (predicateIdentifier != that.predicateIdentifier)
                return false;
            if (!variableGroundings.equals(that.variableGroundings))
                return false;

            return true;
        }

        @Override
        public int hashCode() {
            HashCodeBuilder hashCodeBuilder = new HashCodeBuilder().append(predicateIdentifier);
            variableGroundings.forEach(hashCodeBuilder::append);
            return hashCodeBuilder.toHashCode();
        }
    }
}
