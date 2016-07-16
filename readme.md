# Makina Library

[![Build Status](https://travis-ci.com/eaplatanios/makina.svg?token=VBPxqvcGXTuwbjkVyN68&branch=master)](https://travis-ci.com/eaplatanios/makina)
[![Codecov](https://img.shields.io/codecov/c/token/zQjCSZzyUk/github/eaplatanios/org.platanios.svg)](https://codecov.io/github/eaplatanios/org.platanios?branch=master)

## Estimating Accuracy from Unlabeled Data

All the code related to the "Estimating Accuracy from Unlabeled Data" papers is included in the
`makina.learn.classification.reflection` package. There are currently 4 approaches for unsupervised accuracy estimation:

1. Majority vote based approach, implemented in the `MajorityVoteIntegrator` class.
2. Agreement rates based approach described in [1] and implemented in the `AgreementIntegrator` class.
3. Bayesian approaches described in [2] and implemented in the `BayesianIntegrator`, `CoupledBayesianIntegrator`, and
`HierarchicalCoupledBayesianIntegrator` classes.
<!--4. Logic approach described in [3] and implemented in the `LogicIntegrator` class.-->

All of these classes implement the same abstract class, `Integrator`. In order to construct an `Integrator` object you
need to use the corresponding `Builder` class (it is an inner class in all the integrator implementation classes -- for
more information on the software structure, look into the
[builder design pattern](https://en.wikipedia.org/wiki/Builder_pattern){:target="_blank"}). The following line of code
creates an instance of the `BayesianIntegrator`:

```java
Integrator integrator = new BayesianIntegrator.Builder(predictedData)
								.numberOfBurnInSamples(1000)
								.numberOfThinningSamples(10)
								.numberOfSamples(4000)
								.build();
```

The `predictedData` variable in that line is an instance of `Integrator.Data<Integrator.Data.PredictedInstance>`. Each
`Integrator.Data.PredictedDataInstance` instance can be constructed as follows:

```java
PredictedInstance predictedInstance = new PredictedInstance(id, label, functionId, value);
```

`label` is a `makina.learn.classification.Label` instance that only requires a `String` name for the label to be
constructed, `functionId` is the function/classifier/human ID, `value` is a value in `[0,1]` equal to the probability
of the instance with ID `id` being assigned label `label`.

Finally, an `Integrator.Data<Integrator.Data.PredictedInstance>` can be constructed using a
`List<Integrator.Data.PredictedInstance>`.

### References

1. Emmanouil A. Platanios, Avrim Blum, and Tom Mitchell, *Estimating Accuracy from Unlabeled Data*, Uncertainty in
Artificial Intelligence (UAI), 2014.
2. Emmanouil A. Platanios, Avinava Dubey, and Tom Mitchell, *Estimating Accuracy from Unlabeled Data: A Bayesian
Approach*, International Conference on Machine Learning (ICML), 2016.

<!--## Deep Graphs

Running the Deep Graphs experiments:

```bash
./gradlew run -DXmx100G -DmainClass="makina.experiment.graph.VertexClassificationExperiment" -Dargs="sigmoid 10 1,2,3 10 1 '/home/eplatani/graph/data/WebSpam UK 2007/' 1 '/home/eplatani/graph/results_scala/'"
```-->
