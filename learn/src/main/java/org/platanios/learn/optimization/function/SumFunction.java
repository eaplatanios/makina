package org.platanios.learn.optimization.function;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SumFunction {
    private final int numberOfVariables;
    private final List<int[]> termVariables;
    private final List<AbstractFunction> terms;

    public static class Builder {
        private final int numberOfVariables;
        private final List<int[]> termsVariables = new ArrayList<>();
        private final List<AbstractFunction> terms = new ArrayList<>();

        public Builder(int numberOfVariables) {
            this.numberOfVariables = numberOfVariables;
        }

        public Builder addTerm(AbstractFunction term, int... termVariables) {
            termsVariables.add(termVariables);
            terms.add(term);
            return this;
        }

        public SumFunction build() {
            return new SumFunction(this);
        }
    }

    private SumFunction(Builder builder) {
        numberOfVariables = builder.numberOfVariables;
        termVariables = builder.termsVariables;
        terms = builder.terms;
    }
}
