package org.platanios.learn.classification.reflection;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.classification.Label;
import org.platanios.learn.classification.constraint.Constraint;
import org.platanios.learn.classification.constraint.MutualExclusionConstraint;
import org.platanios.learn.classification.constraint.SubsumptionConstraint;
import org.platanios.learn.data.DataInstance;
import org.platanios.learn.logic.InMemoryLogicManager;
import org.platanios.learn.logic.LogicManager;
import org.platanios.learn.logic.LukasiewiczLogic;
import org.platanios.learn.math.matrix.Vector;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LogicErrorEstimation {
    private static final Logger logger = LogManager.getLogger("Classification / Logic Error Estimation");

    private final LogicManager logicManager;
    private final Set<Label> labels;
    private final Map<Label, Set<Integer>> labelClassifiers;
    private final Map<DataInstance<Vector>, Map<Label, Map<Integer, Double>>> dataSet;
    private final Set<Constraint> constraints;
    private final boolean logProgress;

    private final TuffyLogicIntegrator logicIntegrator;

    public static class Builder {
        private final Set<Label> labels;
        private final Map<Label, Set<Integer>> labelClassifiers;
        private final Map<DataInstance<Vector>, Map<Label, Map<Integer, Double>>> dataSet;

        private LogicManager logicManager = new InMemoryLogicManager(new LukasiewiczLogic());
        private Set<Constraint> constraints = new HashSet<>();
        private boolean logProgress = false;

        public Builder(Map<Label, Set<Integer>> labelClassifiers,
                       Map<DataInstance<Vector>, Map<Label, Map<Integer, Double>>> dataSet) {
            this.labels = labelClassifiers.keySet();
            this.labelClassifiers = labelClassifiers;
            this.dataSet = dataSet;
        }

        public Builder logicManager(LogicManager logicManager) {
            this.logicManager = logicManager;
            return this;
        }

        public Builder addConstraint(MutualExclusionConstraint constraint) {
            constraints.add(constraint);
            return this;
        }

        public Builder addConstraint(SubsumptionConstraint constraint) {
            constraints.add(constraint);
            return this;
        }

        public Builder addConstraints(Set<Constraint> constraints) {
            this.constraints.addAll(constraints);
            return this;
        }

        public Builder logProgress(boolean logProgress) {
            this.logProgress = logProgress;
            return this;
        }

        public LogicErrorEstimation build() {
            return new LogicErrorEstimation(this);
        }
    }

    private LogicErrorEstimation(Builder builder) {
        logicManager = builder.logicManager;
        labels = builder.labels;
        labelClassifiers = builder.labelClassifiers;
        dataSet = builder.dataSet;
        constraints = builder.constraints;
        logProgress = builder.logProgress;
        logicIntegrator =
                new TuffyLogicIntegrator.Builder(labelClassifiers, dataSet)
                        .addConstraints(constraints)
                        .estimateErrorRates(true)
                        .logProgress(logProgress)
                        .workingDirectory("/Users/Anthony/Desktop/tmp/new")
                        .build();
    }

    public ClassifierOutputsIntegrationResults estimateErrorRates() {
        return logicIntegrator.integratePredictions();
    }
}
