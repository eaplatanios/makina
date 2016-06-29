package makina.learn.classification.reflection;

import makina.learn.classification.constraint.MutualExclusionConstraint;
import org.junit.Test;
import makina.learn.classification.Label;
import makina.learn.classification.constraint.Constraint;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LogicIntegratorTest {
    @Test
    public void testSimpleMutualExclusion() {
        Label cityLabel = new Label("city");
        Label animalLabel = new Label("animal");
        Set<Constraint> constraints = new HashSet<>();
        constraints.add(new MutualExclusionConstraint(cityLabel, animalLabel));
        List<Integrator.Data.PredictedInstance> instances = new ArrayList<>();
        // "New York" predictions
        instances.add(new Integrator.Data.PredictedInstance(0, cityLabel, 0, 1.0));
        instances.add(new Integrator.Data.PredictedInstance(0, cityLabel, 1, 1.0));
        instances.add(new Integrator.Data.PredictedInstance(0, cityLabel, 2, 1.0));
        instances.add(new Integrator.Data.PredictedInstance(0, animalLabel, 0, 0.0));
        instances.add(new Integrator.Data.PredictedInstance(0, animalLabel, 1, 1.0));
        instances.add(new Integrator.Data.PredictedInstance(0, animalLabel, 2, 0.0));
        // "London" predictions
        instances.add(new Integrator.Data.PredictedInstance(1, cityLabel, 0, 1.0));
        instances.add(new Integrator.Data.PredictedInstance(1, cityLabel, 1, 1.0));
        instances.add(new Integrator.Data.PredictedInstance(1, cityLabel, 2, 1.0));
        instances.add(new Integrator.Data.PredictedInstance(1, animalLabel, 0, 0.0));
        instances.add(new Integrator.Data.PredictedInstance(1, animalLabel, 1, 1.0));
        instances.add(new Integrator.Data.PredictedInstance(1, animalLabel, 2, 0.0));
        // "Dog" predictions
        instances.add(new Integrator.Data.PredictedInstance(2, cityLabel, 0, 0.0));
        instances.add(new Integrator.Data.PredictedInstance(2, cityLabel, 1, 0.0));
        instances.add(new Integrator.Data.PredictedInstance(2, cityLabel, 2, 0.0));
        instances.add(new Integrator.Data.PredictedInstance(2, animalLabel, 0, 1.0));
        instances.add(new Integrator.Data.PredictedInstance(2, animalLabel, 1, 0.0));
        instances.add(new Integrator.Data.PredictedInstance(2, animalLabel, 2, 1.0));
        // Experiment setup
        Integrator logicErrorEstimation =
                new LogicIntegrator.Builder(new Integrator.Data<>(instances))
                        .addConstraints(constraints)
                        .logProgress(true)
                        .build();
        Integrator.ErrorRates errorRates = logicErrorEstimation.errorRates();
        System.out.println("Finished!");
    }
}
