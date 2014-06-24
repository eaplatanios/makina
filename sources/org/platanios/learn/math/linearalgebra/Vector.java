package org.platanios.learn.math.linearalgebra;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Vector {
    public static double[] add(double[] vector1, double[] vector2) {
        double[] resultVector = new double[vector1.length];
        for (int i = 0; i < vector1.length; i++) {
            resultVector[i] = vector1[i] + vector2[i];
        }
        return resultVector;
    }

    public static double[] subtract(double[] vector1, double[] vector2) {
        double[] resultVector = new double[vector1.length];
        for (int i = 0; i < vector1.length; i++) {
            resultVector[i] = vector1[i] - vector2[i];
        }
        return resultVector;
    }

    public static double[] multiply(double constant, double[] vector) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] *= constant;
        }
        return vector;
    }

    public static double[] multiply(double[][] matrix, double[] vector) {
        double[] resultVector = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            resultVector[i] = computeDotProduct(matrix[i], vector);
        }
        return resultVector;
    }

    public static double computeDotProduct(double[] vector1, double[] vector2) {
        double dotProduct = 0;
        for (int i = 0; i < vector1.length; i++) {
            dotProduct += vector1[i] * vector2[i];
        }
        return dotProduct;
    }

    public static double l2Norm(double[] vector) {
        double l2Norm = 0;
        for (int i = 0; i < vector.length; i++) {
            l2Norm += Math.pow(vector[i], 2);
        }
        return Math.sqrt(l2Norm);
    }
}
