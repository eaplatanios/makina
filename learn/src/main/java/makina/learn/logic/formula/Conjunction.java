package makina.learn.logic.formula;

import makina.learn.logic.LogicManager;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Conjunction extends BranchingFormula {
    public Conjunction(List<Formula> formulas) {
        super(formulas);
    }

    @Override
    public Double evaluate(LogicManager logicManager, Map<Long, Long> variableAssignments) {
        return logicManager.logic().conjunction(
                formulas.stream()
                        .map(formula -> formula.evaluate(logicManager, variableAssignments))
                        .collect(Collectors.toList())
        );
    }

    /**
     * We convert a conjunction into disjunctive normal form (DNF) by using the distributive law of the conjunction.
     * Note that the resulting formula is flattened.
     * // TODO: Need to check for bugs here.
     *
     * @return
     */
    @Override
    public Formula toDisjunctiveNormalForm() {
        List<Formula> components = new ArrayList<>(formulas.size());
        ArrayList<Integer> disjunctionComponents = new ArrayList<>();
        int finalNumberOfComponents = 1;
        for (int componentIndex = 0; componentIndex < formulas.size(); componentIndex++) {
            components.add(formulas.get(componentIndex).toDisjunctiveNormalForm());
            if (components.get(componentIndex) instanceof Disjunction) {
                finalNumberOfComponents *= ((Disjunction) components.get(componentIndex)).getNumberOfComponents();
                disjunctionComponents.add(componentIndex);
            }
        }
        if (disjunctionComponents.size() == 0)
            return new Conjunction(components);
        List<Formula> disjunctiveNormalFormComponents = new ArrayList<>(finalNumberOfComponents);
        int[] indexes = new int[disjunctionComponents.size()];
        for (int finalComponentIndex = 0; finalComponentIndex < finalNumberOfComponents; finalComponentIndex++) {
            for (int j = 0; j < indexes.length; j++) {
                indexes[j]++;
                if (indexes[j] == ((Disjunction) components.get(disjunctionComponents.get(j))).getNumberOfComponents())
                    indexes[j] = 0;
                else
                    break;
            }
            List<Formula> conjunctionComponents = new ArrayList<>(formulas.size());
            for (int componentIndex = 0; componentIndex < conjunctionComponents.size(); componentIndex++) {
                if (components.get(componentIndex) instanceof Disjunction)
                    conjunctionComponents.add(((Disjunction) components.get(componentIndex))
                                                      .getComponent(indexes[componentIndex]));
                else
                    conjunctionComponents.add(components.get(componentIndex));
            }
            disjunctiveNormalFormComponents.add(new Conjunction(conjunctionComponents).toDisjunctiveNormalForm());
        }
        return new Disjunction(disjunctiveNormalFormComponents).flatten();
    }

    public Conjunction flatten() {
        ArrayList<Formula> conjunctionComponents = new ArrayList<>(formulas.size());
        for (Formula formula : formulas) {
            if (formula instanceof Conjunction)
                conjunctionComponents.addAll(((Conjunction) formula).flatten().formulas);
            else
                conjunctionComponents.add(formula);
        }
        return new Conjunction(conjunctionComponents);
    }

    @Override
    public String operatorToString() {
        return "&";
    }
}
