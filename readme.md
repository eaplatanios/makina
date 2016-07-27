# Makina Library

[![Build Status](https://travis-ci.org/eaplatanios/makina.svg?branch=master)](https://travis-ci.org/eaplatanios/makina)
[![Codecov](https://img.shields.io/codecov/c/token/zQjCSZzyUk/github/eaplatanios/org.platanios.svg)](https://codecov.io/github/eaplatanios/org.platanios?branch=master)

## Estimating Accuracy from Unlabeled Data

All the code related to the "Estimating Accuracy from Unlabeled Data" papers is included in the
`makina.learn.classification.reflection` package. There are currently 4 approaches for unsupervised accuracy estimation:

1. Majority vote based approach, implemented in the `MajorityVoteIntegrator` class.
2. Agreement rates based approach described in [1] and implemented in the `AgreementIntegrator` class.
3. Bayesian approaches described in [2] and implemented in the `BayesianIntegrator`, `CoupledBayesianIntegrator`, and
`HierarchicalCoupledBayesianIntegrator` classes.
4. Logic approach described in [3] and implemented in the `LogicIntegrator` class.

All of these classes implement the same abstract class, `Integrator`. In order to construct an `Integrator` object you
need to use the corresponding `Builder` class (it is an inner class in all the integrator implementation classes -- for
more information on the software structure, look into the
<a href="https://en.wikipedia.org/wiki/Builder_pattern" target="_blank">builder design pattern</a>). The following line
of code creates an instance of the `BayesianIntegrator`:

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

### Command-Line Interface

The integrator classes also support a command line interface. Given that the library JAR file is called `makina.jar`, 
the command-line interface can be used as follows:

```bash
usage: java -cp makina.jar makina.learn.classification.reflection.Integrator -d <arg> [-e <arg>] [-h] [-i <arg>] [-m
       <arg>] [-o <arg>]
This command can be used to estimate the accuracies (or equivalently, the error rates) of multiple
functions/classifiers/humans with binary responses, using only unsupervisedor semi-supervised data.
 -d,--dataFile <arg>             Data file location. Supported file extensions are "protobin" and "csv".
 -e,--errorRatesFile <arg>       Output error rates file location. Supported file extensions are "protobin" and "csv".
 -h,--help                       Prints this message.
 -i,--integratedDataFile <arg>   Output integrated data file location. Supported file extensions are "protobin" and
                                 "csv".
 -m,--method <arg>               Method to use (defaults to BI). Currently supported methods include: (i) "MVI", the
                                 majority vote integrator, (ii) "AI", the agreement based integrator, (iii) "BI", the
                                 Bayesian integrator, (iv) "CBI", the coupled Bayesian integrator, (v) "HCBI", the
                                 hierarchical coupled Bayesian integrator, and (vi) "LI", the logic based integrator.
 -o,--options <arg>              Additional options to use for the chosen method. Each method supports a different set
                                 of options, all included in a single string and separated using the ":" character.
                                 Options may not be set by using the "-" character in place of a value. Boolean values
                                 can be set using "1" for "true" and "0" for "false". The specific options allowed for
                                 each method are listed here (the default value for each parameter is shown in
                                 parentheses): (i) "MVI": no options, (ii) "AI": [highest order of agreement rates to
                                 use (all)]:[boolean value indicating to only use even-sized subsets of functions for
                                 agreement rates (1)], (iii) "BI": [number of burn-in samples (4000)]:[number of
                                 thinning samples (10)]:[number of samples (200)]:[labels prior alpha parameter
                                 (1.0)]:[labels prior beta parameter (1.0)]:[error rates prior alpha parameter
                                 (1.0)]:[error rates prior beta parameter (2.0)], (iv) "CBI": [number of burn-in samples
                                 (4000)]:[number of thinning samples (10)]:[number of samples (200)]:[Dirichlet Process
                                 alpha parameter (1.0)]:[labels prior alpha parameter (1.0)]:[labels prior beta
                                 parameter (1.0)]:[error rates prior alpha parameter (1.0)]:[error rates prior beta
                                 parameter (2.0)], (v) "HCBI": [number of burn-in samples (4000)]:[number of thinning
                                 samples (10)]:[number of samples (200)]:[hierarchical Dirichlet Process alpha parameter
                                 (1.0)]:[hierarchical [Dirichlet Process gamma parameter (1.0)]:[labels prior alpha
                                 parameter (1.0)]:[labels prior beta parameter (1.0)]:[error rates prior alpha parameter
                                 (1.0)]:[error rates prior beta parameter (2.0)], (vi) "LI": no options.
For more information refer to the readme file at https://github.com/eaplatanios/makina.
```

An example use of this command is:

```bash
java -cp makina.jar makina.learn.classification.reflection.Integrator -d input_data.csv -e error_rates.csv -i integrated_data.csv -m BI -o 4000:10:200:-:-:-:-
```

The two supported data formats are the Protocol Buffers binary format which can be written to and read from using the 
automatically generated `IntegratorProtos` class in the `makina.learn.classification.reflection` package, and the CSV 
data format. There are three possible CSV format files, one for each different kind of information that can be stored. 
The first one is for predicted data instances (e.g., the input data and the integrated output data) and each line is 
formatted as follows:

```
ID,LABEL,FUNCTION_ID,VALUE
432523,animal,2,0.67
123124,cat,1,0.43
...
```

In this case, `VALUE` corresponds to the probability that the function with ID `FUNCTION_ID` assigns to the data 
instance with ID `ID` for having label `LABEL` assigned to it. The second one is for observed data instances (i.e., 
supervised data instances) and each line is formatted as follows:

```
ID,LABEL,VALUE
432523,animal,1
123124,cat,0
...
```

In this case, `VALUE` corresponds to the boolean observed label (`1` for `true` and `0` for `false`) for label `LABEL` 
and data instance with ID `ID`. Finally, the third kind of information is the error rates information. In that case, 
each line is formatted as follows:

```
LABEL,FUNCTION_ID,VALUE
animal,2,0.12
cat,1,0.05
...
```

In this case, `VALUE` corresponds to the error rate of the function with ID `FUNCTION_ID`, for label `LABEL`.

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
