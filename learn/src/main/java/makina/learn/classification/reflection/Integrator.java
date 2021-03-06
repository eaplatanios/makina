package makina.learn.classification.reflection;

import makina.learn.classification.Label;
import makina.learn.classification.constraint.Constraint;
import makina.learn.classification.constraint.MutualExclusionConstraint;
import makina.learn.classification.constraint.SubsumptionConstraint;
import org.apache.commons.cli.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class Integrator {
    protected static final Logger logger = LogManager.getFormatterLogger("Classification / Reflection / Integrator");

    protected Data<Data.PredictedInstance> data;

    protected ErrorRates errorRates;
    protected Data<Data.PredictedInstance> integratedData;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>> {
        protected abstract T self();

        protected final Data<Data.PredictedInstance> data;

        protected AbstractBuilder(Data<Data.PredictedInstance> data) {
            this.data = data;
        }

        protected AbstractBuilder(String dataFilename) {
            List<Data.PredictedInstance> predictedInstances = new ArrayList<>();
            if(!loadPredictedInstances(dataFilename, predictedInstances))
                throw new RuntimeException("The integrator data could not be loaded from the provided file.");
            data = new Data<>(predictedInstances);
        }

        /**
         *
         *
         * Abstract builder classes for integrators need to override this to process options particular to those
         * integrators.
         *
         * @param   options String containing the integrator options, separated by the ":" character.
         * @return  Current builder instance.
         */
        public T options(String options) {
            return self();
        }
    }

    protected static class Builder extends AbstractBuilder<Builder> {
        protected Builder(Data<Data.PredictedInstance> data) {
            super(data);
        }

        protected Builder(String dataFilename) {
            super(dataFilename);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    protected Integrator(AbstractBuilder<?> builder) {
        data = builder.data;
        errorRates = null;
        integratedData = null;
    }

    public ErrorRates errorRates() {
        return errorRates(false);
    }

    public Data<Data.PredictedInstance> integratedData() {
        return integratedData(false);
    }

    public abstract ErrorRates errorRates(boolean forceComputation);
    public abstract Data<Data.PredictedInstance> integratedData(boolean forceComputation);

    public boolean saveData(String filename) {
        return data != null && savePredictedInstances(filename, data.instances);
    }

    public boolean loadData(String filename) {
        List<Data.PredictedInstance> predictedInstances = new ArrayList<>();
        boolean resultStatus = loadPredictedInstances(filename, predictedInstances);
        data = new Data<>(predictedInstances);
        return resultStatus;
    }

    public boolean saveErrorRates(String filename) {
        String fileExtension = getFilenameExtension(filename);
        switch (fileExtension) {
            case "protobin":
                saveErrorRatesToProtoBin(filename);
                break;
            case "csv":
                saveErrorRatesToCSV(filename);
                break;
            default:
                throw new IllegalArgumentException("Unsupported file extension: " + fileExtension + ".");
        }
        return true;
    }

    public boolean loadErrorRates(String filename) {
        String fileExtension = getFilenameExtension(filename);
        switch (fileExtension) {
            case "protobin":
                loadErrorRatesFromProtoBin(filename);
                break;
            case "csv":
                loadErrorRatesFromCSV(filename);
                break;
            default:
                throw new IllegalArgumentException("Unsupported file extension: " + fileExtension + ".");
        }
        return true;
    }

    public boolean saveErrorRatesToProtoBin(String filename) {
        if (errorRates == null)
            return false;
        try {
            IntegratorProtos.ErrorRates.Builder builder = IntegratorProtos.ErrorRates.newBuilder();
            errorRates.iterator()
                    .forEachRemaining(errorRate -> builder.addErrorRate(
                            IntegratorProtos.ErrorRate.newBuilder()
                                    .setLabel(errorRate.label().name())
                                    .setFunctionId(errorRate.functionId())
                                    .setValue(errorRate.value())
                    ));
            FileOutputStream fileOutputStream = new FileOutputStream(filename);
            builder.build().writeTo(fileOutputStream);
            fileOutputStream.close();
            return true;
        } catch (IOException exception) {
            return false;
        }
    }

    public boolean loadErrorRatesFromProtoBin(String filename) {
        try {
            errorRates = new ErrorRates(
                    IntegratorProtos.ErrorRates.parseFrom(new FileInputStream(filename))
                            .getErrorRateList().stream()
                            .map(errorRate -> new ErrorRates.Instance(
                                    new Label(errorRate.getLabel()),
                                    errorRate.getFunctionId(),
                                    errorRate.getValue()
                            ))
                            .collect(Collectors.toList())
            );
            return true;
        } catch (IOException exception) {
            return false;
        }
    }

    public boolean saveErrorRatesToCSV(String filename) {
        FileWriter writer = null;
        try {
            writer = new FileWriter(filename);
            writer.write("LABEL,FUNCTION_ID,VALUE\n");
            for (ErrorRates.Instance errorRate : errorRates)
                writer.write(errorRate.label().name() + "," +
                                     errorRate.functionId + "," +
                                     errorRate.value() + "\n");
        } catch (IOException exception) {
            logger.error("There was an error while saving error rates to the provided CSV file.", exception);
            return false;
        } finally {
            try {
                if (writer != null) {
                    writer.flush();
                    writer.close();
                }
            } catch (IOException exception) {
                logger.error("There was an error while flushing and closing the CSV file writer.", exception);
            }
        }
        return true;
    }

    public boolean loadErrorRatesFromCSV(String filename) {
        try {
            List<ErrorRates.Instance> errorRatesInstances = new ArrayList<>();
            Files.lines(Paths.get(filename)).forEach(line -> {
                if (!line.equals("LABEL,FUNCTION_ID,VALUE")) {
                    String[] lineParts = line.split(",");
                    errorRatesInstances.add(new ErrorRates.Instance(new Label(lineParts[0]),
                                                                    Integer.parseInt(lineParts[1]),
                                                                    Double.parseDouble(lineParts[2])));
                }
            });
            errorRates = new ErrorRates(errorRatesInstances);
            return true;
        } catch (IOException exception) {
            logger.error("There was an error while loading error rates from the provided CSV file.", exception);
            return false;
        }
    }

    public boolean saveIntegratedData(String filename) {
        return integratedData != null && savePredictedInstances(filename, integratedData.instances);
    }

    public boolean loadIntegratedData(String filename) {
        List<Data.PredictedInstance> predictedInstances = new ArrayList<>();
        boolean resultStatus = loadPredictedInstances(filename, predictedInstances);
        integratedData = new Data<>(predictedInstances);
        return resultStatus;
    }

    public static boolean savePredictedInstances(String filename, List<Data.PredictedInstance> predictedInstances) {
        String fileExtension = getFilenameExtension(filename);
        switch (fileExtension) {
            case "protobin":
                savePredictedInstancesToProtoBin(filename, predictedInstances);
                break;
            case "csv":
                savePredictedInstancesToCSV(filename, predictedInstances);
                break;
            default:
                throw new IllegalArgumentException("Unsupported file extension: " + fileExtension + ".");
        }
        return true;
    }

    public static boolean loadPredictedInstances(String filename, List<Data.PredictedInstance> predictedInstances) {
        String fileExtension = getFilenameExtension(filename);
        switch (fileExtension) {
            case "protobin":
                loadPredictedInstancesFromProtoBin(filename, predictedInstances);
                break;
            case "csv":
                loadPredictedInstancesFromCSV(filename, predictedInstances);
                break;
            default:
                throw new IllegalArgumentException("Unsupported file extension: " + fileExtension + ".");
        }
        return true;
    }

    public static boolean savePredictedInstancesToProtoBin(String filename,
                                                           List<Data.PredictedInstance> predictedInstances) {
        try {
            IntegratorProtos.PredictedInstances.Builder builder = IntegratorProtos.PredictedInstances.newBuilder();
            predictedInstances.iterator()
                    .forEachRemaining(predictedInstance -> builder.addPredictedInstance(
                            IntegratorProtos.PredictedInstance.newBuilder()
                                    .setId(predictedInstance.id())
                                    .setLabel(predictedInstance.label().name())
                                    .setFunctionId(predictedInstance.functionId())
                                    .setValue(predictedInstance.value())
                    ));
            FileOutputStream fileOutputStream = new FileOutputStream(filename);
            builder.build().writeTo(fileOutputStream);
            fileOutputStream.close();
            return true;
        } catch (IOException exception) {
            logger.error("There was an error while saving predicted instances to the provided proto binary file.",
                         exception);
            return false;
        }
    }

    public static boolean loadPredictedInstancesFromProtoBin(String filename,
                                                             List<Data.PredictedInstance> predictedInstances) {
        try {
            predictedInstances.addAll(IntegratorProtos.PredictedInstances.parseFrom(new FileInputStream(filename))
                    .getPredictedInstanceList().stream()
                    .map(predictedInstance -> new Data.PredictedInstance(
                            predictedInstance.getId(),
                            new Label(predictedInstance.getLabel()),
                            predictedInstance.getFunctionId(),
                            predictedInstance.getValue()
                    ))
                    .collect(Collectors.toList()));
            return true;
        } catch (IOException exception) {
            logger.error("There was an error while loading predicted instances from the provided proto binary file.",
                         exception);
            return false;
        }
    }

    public static boolean savePredictedInstancesToCSV(String filename,
                                                      List<Data.PredictedInstance> predictedInstances) {
        FileWriter writer = null;
        try {
            writer = new FileWriter(filename);
            writer.write("ID,LABEL,FUNCTION_ID,VALUE\n");
            for (Data.PredictedInstance predictedInstance : predictedInstances)
                writer.write(predictedInstance.id() + "," +
                                     predictedInstance.label().name() + "," +
                                     predictedInstance.functionId + "," +
                                     predictedInstance.value() + "\n");
        } catch (IOException exception) {
            logger.error("There was an error while saving predicted instances to the provided CSV file.", exception);
            return false;
        } finally {
            try {
                if (writer != null) {
                    writer.flush();
                    writer.close();
                }
            } catch (IOException exception) {
                logger.error("There was an error while flushing and closing the CSV file writer.", exception);
            }
        }
        return true;
    }

    public static boolean loadPredictedInstancesFromCSV(String filename,
                                                        List<Data.PredictedInstance> predictedInstances) {
        try {
            Files.lines(Paths.get(filename)).forEach(line -> {
                if (!line.equals("ID,LABEL,FUNCTION_ID,VALUE")) {
                    String[] lineParts = line.split(",");
                    predictedInstances.add(new Data.PredictedInstance(Integer.parseInt(lineParts[0]),
                                                                      new Label(lineParts[1]),
                                                                      Integer.parseInt(lineParts[2]),
                                                                      Double.parseDouble(lineParts[3])));
                }
            });
            return true;
        } catch (IOException exception) {
            logger.error("There was an error while loading predicted instances from the provided CSV file.", exception);
            return false;
        }
    }

    public static boolean saveObservedInstances(String filename, List<Data.ObservedInstance> observedInstances) {
        String fileExtension = getFilenameExtension(filename);
        switch (fileExtension) {
            case "protobin":
                saveObservedInstancesToProtoBin(filename, observedInstances);
                break;
            case "csv":
                saveObservedInstancesToCSV(filename, observedInstances);
                break;
            default:
                throw new IllegalArgumentException("Unsupported file extension: " + fileExtension + ".");
        }
        return true;
    }

    public static boolean loadObservedInstances(String filename, List<Data.ObservedInstance> observedInstances) {
        String fileExtension = getFilenameExtension(filename);
        switch (fileExtension) {
            case "protobin":
                loadObservedInstancesFromProtoBin(filename, observedInstances);
                break;
            case "csv":
                loadObservedInstancesFromCSV(filename, observedInstances);
                break;
            default:
                throw new IllegalArgumentException("Unsupported file extension: " + fileExtension + ".");
        }
        return true;
    }

    public static boolean saveObservedInstancesToProtoBin(String filename,
                                                          List<Data.ObservedInstance> observedInstances) {
        try {
            IntegratorProtos.ObservedInstances.Builder builder = IntegratorProtos.ObservedInstances.newBuilder();
            observedInstances.iterator()
                    .forEachRemaining(observedInstance -> builder.addObservedInstance(
                            IntegratorProtos.ObservedInstance.newBuilder()
                                    .setId(observedInstance.id())
                                    .setLabel(observedInstance.label().name())
                                    .setValue(observedInstance.value())
                    ));
            FileOutputStream fileOutputStream = new FileOutputStream(filename);
            builder.build().writeTo(fileOutputStream);
            fileOutputStream.close();
            return true;
        } catch (IOException exception) {
            logger.error("There was an error while saving observed instances to the provided proto binary file.",
                         exception);
            return false;
        }
    }

    public static boolean loadObservedInstancesFromProtoBin(String filename,
                                                            List<Data.ObservedInstance> observedInstances) {
        try {
            observedInstances.addAll(IntegratorProtos.ObservedInstances.parseFrom(new FileInputStream(filename))
                    .getObservedInstanceList().stream()
                    .map(observedInstance -> new Data.ObservedInstance(
                            observedInstance.getId(),
                            new Label(observedInstance.getLabel()),
                            observedInstance.getValue()
                    ))
                    .collect(Collectors.toList()));
            return true;
        } catch (IOException exception) {
            logger.error("There was an error while loading observed instances from the provided proto binary file.",
                         exception);
            return false;
        }
    }

    public static boolean saveObservedInstancesToCSV(String filename, List<Data.ObservedInstance> observedInstances) {
        FileWriter writer = null;
        try {
            writer = new FileWriter(filename);
            writer.write("ID,LABEL,VALUE\n");
            for (Data.ObservedInstance observedInstance : observedInstances)
                writer.write(observedInstance.id() + "," +
                                     observedInstance.label().name() + "," +
                                     (observedInstance.value() ? "1" : "0") + "\n");
        } catch (IOException exception) {
            logger.error("There was an error while saving observed instances to the provided CSV file.", exception);
            return false;
        } finally {
            try {
                if (writer != null) {
                    writer.flush();
                    writer.close();
                }
            } catch (IOException exception) {
                logger.error("There was an error while flushing and closing the CSV file writer.", exception);
            }
        }
        return true;
    }

    public static boolean loadObservedInstancesFromCSV(String filename,
                                                       List<Data.ObservedInstance> observedInstances) {
        try {
            Files.lines(Paths.get(filename)).forEach(line -> {
                if (!line.equals("ID,LABEL,VALUE")) {
                    String[] lineParts = line.split(",");
                    observedInstances.add(new Data.ObservedInstance(Integer.parseInt(lineParts[0]),
                                                                    new Label(lineParts[1]),
                                                                    lineParts[2].equals("1")));
                }
            });
            return true;
        } catch (IOException exception) {
            logger.error("There was an error while loading observed instances from the provided CSV file.", exception);
            return false;
        }
    }

    private static String getFilenameExtension(String filename) {
        String[] filenameParts = filename.split("\\.");
        return filenameParts[filenameParts.length - 1];
    }

    public static boolean saveConstraints(String filename, Set<Constraint> constraints) {
        FileWriter writer = null;
        try {
            writer = new FileWriter(filename);
            for (Constraint constraint : constraints)
                if (constraint instanceof MutualExclusionConstraint)
                    writer.write("!" + String.join(",", ((MutualExclusionConstraint) constraint).getLabels().stream()
                            .map(Label::name).collect(Collectors.toList())) + "\n");
                else if (constraint instanceof SubsumptionConstraint)
                    writer.write(((SubsumptionConstraint) constraint).getParentLabel().name() + " -> " +
                                         ((SubsumptionConstraint) constraint).getChildLabel().name() + "\n");
                else
                    logger.warn("Unsupported constraint type encountered and will not be saved.");
        } catch (IOException exception) {
            logger.error("There was an error while saving observed instances to the provided file.", exception);
            return false;
        } finally {
            try {
                if (writer != null) {
                    writer.flush();
                    writer.close();
                }
            } catch (IOException exception) {
                logger.error("There was an error while flushing and closing the file writer.", exception);
            }
        }
        return true;
    }

    public static Set<Constraint> loadConstraints(String filename) {
        Set<Constraint> constraints = new HashSet<>();
        try {
            Files.newBufferedReader(Paths.get(filename)).lines().forEach(line -> {
                if (line.startsWith("!")) {
                    constraints.add(new MutualExclusionConstraint(Arrays.stream(line.substring(1).split(","))
                                                                          .map(Label::new)
                                                                          .collect(Collectors.toSet())));
                } else {
                    String[] lineParts = line.split(" -> ");
                    String[] childrenLabels = lineParts[1].split(",");
                    for (String childLabel : childrenLabels)
                        constraints.add(new SubsumptionConstraint(new Label(lineParts[0]), new Label(childLabel)));
                }
            });
        } catch (IOException e) {
            throw new IllegalArgumentException("There was a problem with the provided constraints file.");
        }
        return constraints;
    }

    public static class Data<T extends Data.Instance> implements Iterable<T> {
        private final List<T> instances;

        public Data(List<T> instances) {
            this.instances = instances;
        }

        public int size() {
            return instances.size();
        }

        public T get(int index) {
            if (index >= instances.size())
                throw new IllegalArgumentException("The provided instance index is out of bounds.");
            return instances.get(index);
        }

        @Override
        public Iterator<T> iterator() {
            return instances.iterator();
        }

        public Stream<T> stream() {
            return instances.stream();
        }

        public static abstract class Instance {
            private final int id;
            private final Label label;

            public Instance(int id, Label label) {
                this.id = id;
                this.label = label;
            }

            public int id() {
                return id;
            }

            public Label label() {
                return label;
            }
        }

        public static class ObservedInstance extends Instance {
            private final boolean value;

            public ObservedInstance(int id, Label label, boolean value) {
                super(id, label);
                this.value = value;
            }

            public boolean value() {
                return value;
            }
        }

        public static class PredictedInstance extends Instance {
            private final int functionId;
            private final double value;

            public PredictedInstance(int id, Label label, double value) {
                super(id, label);
                this.functionId = -1;
                this.value = value;
            }

            public PredictedInstance(int id, Label label, int functionId, double value) {
                super(id, label);
                this.functionId = functionId;
                this.value = value;
            }

            public int functionId() {
                return functionId;
            }

            public double value() {
                return value;
            }
        }
    }

    public static class ErrorRates implements Iterable<ErrorRates.Instance> {
        private final List<Instance> instances;

        public ErrorRates(List<Instance> instances) {
            this.instances = instances;
        }

        public int size() {
            return instances.size();
        }

        public Instance get(int index) {
            if (index >= instances.size())
                throw new IllegalArgumentException("The provided instance index is out of bounds.");
            return instances.get(index);
        }

        @Override
        public Iterator<Instance> iterator() {
            return instances.iterator();
        }

        public Stream<Instance> stream() {
            return instances.stream();
        }

        public static class Instance {
            private final Label label;
            private final int functionId;
            private final double value;

            public Instance(Label label, int functionId, double value) {
                this.label = label;
                this.functionId = functionId;
                this.value = value;
            }

            public Label label() {
                return label;
            }

            public int functionId() {
                return functionId;
            }

            public double value() {
                return value;
            }
        }
    }

    public static void main(String[] args) {
        Options commandLineOptions = new Options();
        commandLineOptions.addOption(Option.builder("d").longOpt("predictedDataFile")
                                             .desc("Predicted data file location. Supported file extensions are " +
                                                           "\"protobin\" and \"csv\".")
                                             .hasArg().required().build());
        commandLineOptions.addOption(Option.builder("od").longOpt("observedDataFile")
                                             .desc("Observed data file location. Supported file extensions are " +
                                                           "\"protobin\" and \"csv\".")
                                             .hasArg().build());
        commandLineOptions.addOption(Option.builder("c").longOpt("constraintsFile")
                                             .desc("Constraints file location.")
                                             .hasArg().build());
        commandLineOptions.addOption(Option.builder("m").longOpt("method")
                                             .desc("Method to use (defaults to BI). Currently supported methods " +
                                                           "include: (i) \"MVI\", the majority vote integrator, (ii) " +
                                                           "\"AI\", the agreement based integrator, (iii) \"BI\", " +
                                                           "the Bayesian integrator, (iv) \"CBI\", the coupled " +
                                                           "Bayesian integrator, (v) \"HCBI\", the hierarchical " +
                                                           "coupled Bayesian integrator, and (vi) \"LI\", the logic " +
                                                           "based integrator.")
                                             .hasArg().build());
        commandLineOptions.addOption(Option.builder("o").longOpt("options")
                                             .desc("Additional options to use for the chosen method. Each method " +
                                                           "supports a different set of options, all included in a " +
                                                           "single string and separated using the \":\" character. " +
                                                           "Options may not be set by using the \"-\" character in " +
                                                           "place of a value. Boolean values can be set using \"1\" " +
                                                           "for \"true\" and \"0\" for \"false\". The specific " +
                                                           "options allowed for each method are listed here (the " +
                                                           "default value for each parameter is shown in " +
                                                           "parentheses): (i) \"MVI\": no options, (ii) \"AI\": " +
                                                           "[highest order of agreement rates to use (all)]:[boolean " +
                                                           "value indicating to only use even-sized subsets of " +
                                                           "functions for agreement rates (1)], (iii) \"BI\": " +
                                                           "[number of burn-in samples (4000)]:" +
                                                           "[number of thinning samples (10)]:" +
                                                           "[number of samples (200)]:" +
                                                           "[labels prior alpha parameter (1.0)]:" +
                                                           "[labels prior beta parameter (1.0)]:" +
                                                           "[error rates prior alpha parameter (1.0)]:" +
                                                           "[error rates prior beta parameter (2.0)], (iv) \"CBI\": " +
                                                           "[number of burn-in samples (4000)]:" +
                                                           "[number of thinning samples (10)]:" +
                                                           "[number of samples (200)]:" +
                                                           "[Dirichlet Process alpha parameter (1.0)]:" +
                                                           "[labels prior alpha parameter (1.0)]:" +
                                                           "[labels prior beta parameter (1.0)]:" +
                                                           "[error rates prior alpha parameter (1.0)]:" +
                                                           "[error rates prior beta parameter (2.0)], (v) \"HCBI\": " +
                                                           "[number of burn-in samples (4000)]:" +
                                                           "[number of thinning samples (10)]:" +
                                                           "[number of samples (200)]:" +
                                                           "[hierarchical Dirichlet Process alpha parameter (1.0)]:" +
                                                           "[hierarchical [Dirichlet Process gamma parameter (1.0)]:" +
                                                           "[labels prior alpha parameter (1.0)]:" +
                                                           "[labels prior beta parameter (1.0)]:" +
                                                           "[error rates prior alpha parameter (1.0)]:" +
                                                           "[error rates prior beta parameter (2.0)], (vi) \"LI\": " +
                                                           "no options.")
                                             .hasArg().build());
        commandLineOptions.addOption(Option.builder("e").longOpt("errorRatesFile")
                                             .desc("Output error rates file location. Supported file extensions are " +
                                                           "\"protobin\" and \"csv\".")
                                             .hasArg().build());
        commandLineOptions.addOption(Option.builder("i").longOpt("integratedDataFile")
                                             .desc("Output integrated data file location. Supported file extensions " +
                                                           "are \"protobin\" and \"csv\".")
                                             .hasArg().build());
        commandLineOptions.addOption(Option.builder("s").longOpt("randomSeed")
                                             .desc("Seed to use for the random number generator (only relevant if " +
                                                           "the chosen integrator uses a random number generator).")
                                             .hasArg().build());
        commandLineOptions.addOption(Option.builder("h").longOpt("help").desc("Prints this message.").build());
        // Parse the command-line arguments
        CommandLineParser parser = new DefaultParser();
        HelpFormatter helpFormatter = new HelpFormatter();
        helpFormatter.setWidth(120);
        try {
            CommandLine line = parser.parse(commandLineOptions, args);
            if (line.hasOption("h"))
                helpFormatter.printHelp(
                        "java -cp makina.jar makina.learn.classification.reflection.Integrator",
                        "This command can be used to estimate the accuracies (or equivalently, the error rates) of " +
                                "multiple functions/classifiers/humans with binary responses, using only unsupervised" +
                                "or semi-supervised data.",
                        commandLineOptions,
                        "For more information refer to the readme file at https://github.com/eaplatanios/makina.",
                        true
                );
            String predictedDataFile = line.getOptionValue("d");
            String observedDataFile = line.getOptionValue("od", null);
            String constraintsFile = line.getOptionValue("c", null);
            Set<Constraint> constraints = new HashSet<>();
            if (constraintsFile != null)
                constraints.addAll(loadConstraints(constraintsFile));
            String method = line.getOptionValue("m", "BI");
            String options = line.getOptionValue("o", "");
            String seedString = line.getOptionValue("s", null);
            Long seed = null;
            if (seedString != null)
                seed = Long.parseLong(seedString);
            String errorRatesFile = line.getOptionValue("e", null);
            String integratedDataFile = line.getOptionValue("i", null);
            if (errorRatesFile == null && integratedDataFile == null) {
                logger.error("No output files where specified. Either the error rates or the integrated data output " +
                                     "files (i.e., '-e' and '-i' command-line options, respectively) need to be " +
                                     "specified.");
                return;
            }
            Integrator integrator;
            switch (method) {
                case "MVI":
                    integrator = new MajorityVoteIntegrator.Builder(predictedDataFile).options(options).build();
                    break;
                case "AI":
                    if (observedDataFile != null)
                        integrator = new AgreementIntegrator.Builder(predictedDataFile, observedDataFile)
                                .options(options).build();
                    else
                        integrator = new AgreementIntegrator.Builder(predictedDataFile).options(options).build();
                    break;
                case "BI":
//                    if (observedDataFile != null)
//                        integrator = new BayesianIntegrator.Builder(predictedDataFile, observedDataFile).seed(seed)
//                                .options(options).build();
//                    else
                    integrator = new BayesianIntegrator.Builder(predictedDataFile).seed(seed).options(options).build();
                    break;
                case "CBI":
//                    if (observedDataFile != null)
//                        integrator = new CoupledBayesianIntegrator.Builder(predictedDataFile, observedDataFile)
//                                .seed(seed).options(options).build();
//                    else
                    integrator = new CoupledBayesianIntegrator.Builder(predictedDataFile).seed(seed).options(options).build();
                    break;
                case "HCBI":
//                    if (observedDataFile != null)
//                        integrator = new HierarchicalCoupledBayesianIntegrator.Builder(predictedDataFile,
//                                                                                       observedDataFile).seed(seed)
//                                .options(options).build();
//                    else
                    integrator = new HierarchicalCoupledBayesianIntegrator.Builder(predictedDataFile).seed(seed).options(options).build();
                    break;
                case "LI":
                    if (observedDataFile != null)
                        integrator = new LogicIntegrator.Builder(predictedDataFile, observedDataFile)
                                .addConstraints(constraints).options(options).build();
                    else
                        integrator = new LogicIntegrator.Builder(predictedDataFile)
                                .addConstraints(constraints).options(options).build();
                    break;
                default:
                    throw new IllegalArgumentException("Unsupported method name provided.");
            }
            integrator.errorRates();
            integrator.integratedData();
            if (errorRatesFile != null)
                integrator.saveErrorRates(errorRatesFile);
            if (integratedDataFile != null)
                integrator.saveIntegratedData(integratedDataFile);
        } catch (ParseException exception) {
            helpFormatter.printHelp(
                    "java -cp makina.jar makina.learn.classification.reflection.Integrator",
                    "This command can be used to estimate the accuracies (or equivalently, the error rates) of " +
                            "multiple functions/classifiers/humans with binary responses, using only unsupervised" +
                            "or semi-supervised data.",
                    commandLineOptions,
                    "For more information refer to the readme file at https://github.com/eaplatanios/makina.",
                    true
            );
        }
    }
}
