[![Build Status](https://travis-ci.com/eaplatanios/org.platanios.svg?token=VBPxqvcGXTuwbjkVyN68&branch=master)](https://travis-ci.com/eaplatanios/org.platanios)
[![Codecov](https://img.shields.io/codecov/c/token/zQjCSZzyUk/github/eaplatanios/org.platanios.svg)](https://codecov.io/github/eaplatanios/org.platanios?branch=master)

# Makina Library

## Estimating Accuracy from Unlabeled Data



## Deep Graphs

Running the Deep Graphs experiments:

```bash
./gradlew run -DXmx100G -DmainClass="makina.experiment.graph.VertexClassificationExperiment" -PappArgs="['sigmoid','10','1,2,3','10','1','/home/eplatani/graph/data/WebSpam UK 2007/','1','/home/eplatani/graph/results_scala/']"
```
