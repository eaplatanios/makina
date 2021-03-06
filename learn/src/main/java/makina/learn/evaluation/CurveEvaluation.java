package makina.learn.evaluation;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.AreaChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;
import javafx.util.converter.NumberStringConverter;
import makina.learn.data.DataSet;
import makina.learn.data.PredictedDataInstance;
import makina.utilities.MathUtilities;
import makina.math.matrix.Vector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class CurveEvaluation<T extends Vector, S> {
    protected static final double epsilon = MathUtilities.computeMachineEpsilonDouble();

    protected final Map<String, Curve> curves = new HashMap<>();
    protected final Map<String, Double> areaUnderCurves = new HashMap<>();

    protected CurveEvaluation() { }

    // TODO: It is maybe better to have a getList() method in the data set classes.
    // TODO: Assumes that the labels of all predicted data instances are positive.
    public void addResult(String name,
                          DataSet<PredictedDataInstance<T, S>> predictions,
                          Function<PredictedDataInstance<T, S>, Boolean> groundTruth) {
        List<PredictedDataInstance<T, S>> predictionsList = new ArrayList<>();
        predictions.forEach(predictionsList::add);
        addResult(name, predictionsList, groundTruth);
    }

    public abstract void addResult(String name,
                                   List<PredictedDataInstance<T, S>> predictions,
                                   Function<PredictedDataInstance<T, S>, Boolean> groundTruth);

    protected abstract String getPlotTitle();
    protected abstract String getVerticalAxisName();
    protected abstract String getHorizontalAxisName();

    public List<Curve> getCurves() {
        return curves.values().stream().collect(Collectors.toList());
    }

    public Curve getCurve(String name) {
        return curves.get(name);
    }

    public List<Double> getAreaUnderCurves() {
        return areaUnderCurves.values().stream().collect(Collectors.toList());
    }

    public Double getAreaUnderCurve(String name) {
        return areaUnderCurves.get(name);
    }

    public void plotCurves() {
        Plot.setPlotTitle(getPlotTitle());
        Plot.setHorizontalAxisName(getHorizontalAxisName());
        Plot.setVerticalAxisName(getVerticalAxisName());
        Plot.addCurves(getCurves(), getAreaUnderCurves());
        Application.launch(Plot.class);
    }

    public static class CurvePoint {
        private final double horizontalAxisValue;
        private final double verticalAxisValue;

        public CurvePoint(double horizontalAxisValue, double verticalAxisValue) {
            this.horizontalAxisValue = horizontalAxisValue;
            this.verticalAxisValue = verticalAxisValue;
        }

        public double getHorizontalAxisValue() {
            return horizontalAxisValue;
        }

        public double getVerticalAxisValue() {
            return verticalAxisValue;
        }
    }

    public static class Curve {
        private final String name;
        private final List<CurvePoint> points;

        public Curve(String name, List<CurvePoint> points) {
            this.name = name;
            this.points = points;
        }

        public String getName() {
            return name;
        }

        public List<CurvePoint> getPoints() {
            return points;
        }
    }

    public static class Plot extends Application {
        private static final List<Curve> curves = new ArrayList<>();
        private static final List<Double> areaUnderCurves = new ArrayList<>();

        private static String plotTitle;
        private static String horizontalAxisName;
        private static String verticalAxisName;

        private final NumberAxis xAxis = new NumberAxis(); //(horizontalAxisName, 0, 1, 0.2);
        private final NumberAxis yAxis = new NumberAxis(); //(verticalAxisName, 0, 1, 0.2);
        private final AreaChart<Number, Number> areaChart = new AreaChart<>(xAxis, yAxis);

        public Plot() { }

        public static void setPlotTitle(String plotTitle) {
            Plot.plotTitle = plotTitle;
        }

        public static void setHorizontalAxisName(String horizontalAxisName) {
            Plot.horizontalAxisName = horizontalAxisName;
        }

        public static void setVerticalAxisName(String verticalAxisName) {
            Plot.verticalAxisName = verticalAxisName;
        }

        private static void addCurves(List<Curve> curves, List<Double> areaUnderCurves) {
            Plot.curves.addAll(curves);
            Plot.areaUnderCurves.addAll(areaUnderCurves);
        }

        @Override
        public void start(Stage stage) {
            stage.setTitle(plotTitle);
            areaChart.setTitle(plotTitle);
            areaChart.setCreateSymbols(false);
            xAxis.setTickLabelFormatter(new NumberStringConverter("0.0"));
            yAxis.setTickLabelFormatter(new NumberStringConverter("0.0"));
            for (int curveIndex = 0; curveIndex < curves.size(); curveIndex++) {
                XYChart.Series<Number, Number> curveSeries = new XYChart.Series<>();
                curveSeries.setName(curves.get(curveIndex).name
                                            + " - AUC: " + String.format("%.4f", areaUnderCurves.get(curveIndex)));
                for (CurvePoint point : curves.get(curveIndex).points)
                    curveSeries.getData().add(new XYChart.Data<>(point.horizontalAxisValue, point.verticalAxisValue));
                areaChart.getData().add(curveSeries);
            }
            Scene scene = new Scene(areaChart, 350, 400);
            scene.getStylesheets().add("makina/learn/evaluation/CurveEvaluation.Plot.css");
            stage.setScene(scene);
            stage.show();
        }
    }
}
