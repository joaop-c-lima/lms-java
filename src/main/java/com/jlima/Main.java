package com.jlima;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class Main {
    public static void generateCsvDataset(int dimension, int numSamples, String filePath) {
        Random rand = new Random();
        double[] weights = new double[dimension];

        // Generate random weights for the linear model
        for (int i = 0; i < dimension; i++) {
            weights[i] = rand.nextDouble() * 10 - 5; // weights in [-5, 5]
        }

        try (FileWriter writer = new FileWriter(filePath)) {
            // Write header
            for (int i = 0; i < dimension; i++) {
                writer.append("x").append(String.valueOf(i)).append(",");
            }
            writer.append("y\n");

            // Generate data
            for (int sample = 0; sample < numSamples; sample++) {
                double[] x = new double[dimension];
                double y = 0.0;

                // Generate features and compute target
                for (int i = 0; i < dimension; i++) {
                    x[i] = rand.nextDouble() * 20 - 10; // features in [-10, 10]
                    y += x[i] * weights[i];
                    writer.append(String.format("%.4f", x[i]).replace(",",".")).append(",");
                }

                // Add small Gaussian noise
                y += rand.nextGaussian() * 0.5;

                writer.append(String.format("%.4f", y).replace(",",".")).append("\n");
            }

            System.out.println("CSV dataset generated: " + filePath);
        } catch (IOException e) {
            System.err.println("Failed to write CSV: " + e.getMessage());
        }
    }

    public static void main(String[] args) throws IOException {
        String path = "";
        generateCsvDataset(10, 10000, path+ "regression_dataset.csv");
        LMS lms = new LMS(path+"regression_dataset.csv", 100, "y", null, 1000, 50, 0.001, true);
        lms.train();
        lms.exportLearningCurve(path + "learning_curve.png");
    }


}
