package com.jlima;

import org.ejml.simple.SimpleMatrix;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class DatasetLoader {

    private final List<String> labels;
    private final int targetColumn;
    private final int dimension;

    public DatasetLoader(String headerLine, String targetColumnName) {
        this.labels = Arrays.asList(headerLine.trim().split(","));
        labels.replaceAll(String::trim);

        this.targetColumn = labels.indexOf(targetColumnName);
        if (this.targetColumn == -1) {
            throw new IllegalArgumentException("Target column not found: " + targetColumnName);
        }

        this.dimension = labels.size() - 1;
    }

    public List<Sample> loadDataset(String filePath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String header = reader.readLine();
            if (header == null) throw new IOException("Empty file: " + filePath);

            List<String> actualLabels = Arrays.asList(header.split(","));
            actualLabels.replaceAll(String::trim);

            if (!actualLabels.equals(labels)) {
                throw new IllegalArgumentException("Header mismatch in file: " + filePath);
            }

            List<Sample> samples = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                samples.add(extractSample(line));
            }
            return samples;
        }
    }

    public Sample extractSample(String line) {
        String[] tokens = line.trim().split(",");
        if (tokens.length != labels.size()) {
            throw new IllegalArgumentException("Expected " + labels.size() +
                    " columns but got " + tokens.length);
        }

        double[] x = new double[dimension + 1];
        x[0] = 1.0; // Bias
        double target = 0.0;
        int xi = 1;

        for (int i = 0; i < tokens.length; i++) {
            double value;
            try {
                value = Double.parseDouble(tokens[i].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid numeric value: " + tokens[i]);
            }

            if (i == targetColumn) {
                target = value;
            } else {
                x[xi++] = value;
            }
        }

        return new Sample(new SimpleMatrix(x), target);
    }

    public List<String> getLabels() {
        return labels;
    }

    public int getTargetColumn() {
        return targetColumn;
    }

    public int getDimension() {
        return dimension;
    }
}
