package org.platanios.learn.classification;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.serialization.UnsafeSerializationUtilities;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.*;
import java.sql.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class FeatureMapMariaDB<T extends Vector> extends FeatureMap<T> {
    private static final Logger logger = LogManager.getLogger("Classification / MariaDB Storage");

    private final Connection connection;

    protected FeatureMapMariaDB(int numberOfViews) {
        super(numberOfViews);
        try {
            connection = DriverManager.getConnection("jdbc:mariadb://localhost/", "root", null);
        } catch (SQLException e) {
            logger.error("Could not connect to the default database server!", e);
            throw new RuntimeException(e);
        }
    }

    protected FeatureMapMariaDB(int numberOfViews, String host, String username, String password) {
        super(numberOfViews);
        try {
            connection = DriverManager.getConnection(host, username, password);
        } catch (SQLException e) {
            logger.error("Could not connect to the default database!", e);
            throw new RuntimeException(e);
        }
    }

    @Override
    @SuppressWarnings("unchecked")
    public void loadFeatureMap(InputStream inputStream) {
        try {
            numberOfViews = UnsafeSerializationUtilities.readInt(inputStream);
            int numberOfKeys = UnsafeSerializationUtilities.readInt(inputStream);
            for (int i = 0; i < numberOfKeys; i++) {
                String name = UnsafeSerializationUtilities.readString(inputStream, 1024);
                List<T> features = new ArrayList<>();
                for (int view = 0; view < numberOfViews; view++)
                    features.add((T) Vectors.build(inputStream));
                insertFeatures(name, features);
            }
            inputStream.close();
            logger.debug("Loaded the feature map from the provided input stream.");
        } catch (Exception e) {
            logger.error("An exception was thrown while loading the feature map from the provided input stream.", e);
            throw new RuntimeException(e);
        }
    }

    @Override
    public void loadFeatureMap(Connection databaseConnection) {
        throw new NotImplementedException();
    }

    @Override
    public void addSingleViewFeatureMappings(String name, T features, int view) {
        try {
            insertFeatures(name, features, view);
        } catch (SQLException e) {
            logger.error("Could not insert single view features for name '" + name + "' and view " + view + ".", e);
            throw new RuntimeException(e);
        }
    }

    @Override
    public void addSingleViewFeatureMappings(Map<String, T> featureMappings, int view) {
        try {
            insertFeatures(featureMappings, view);
        } catch (SQLException e) {
            logger.error("Could not insert single view feature mappings for view " + view + ".", e);
            throw new RuntimeException(e);
        }
    }

    @Override
    public void addFeatureMappings(String name, List<T> features) {
        try {
            insertFeatures(name, features);
        } catch (SQLException e) {
            logger.error("Could not insert multiple views features for name '" + name + "'.", e);
            throw new RuntimeException(e);
        }
    }

    @Override
    public void addFeatureMappings(Map<String, List<T>> featureMappings) {
        try {
            insertFeatures(featureMappings);
        } catch (SQLException e) {
            logger.error("Could not insert multiple views features for multiple names.", e);
            throw new RuntimeException(e);
        }
    }

    @Override
    public T getSingleViewFeatureVector(String name, int view) {
        try {
            return selectFeatures(name, view);
        } catch (SQLException|IOException e) {
            logger.error("Could not get single view features for name '" + name + "' and view " + view + ".", e);
            throw new RuntimeException(e);
        }
    }

    @Override
    public Map<String, T> getSingleViewFeatureVectors(List<String> names, int view) {
        try {
            return selectFeatures(names, view);
        } catch (SQLException|IOException e) {
            logger.error("Could not get single view features for the given list of names and view " + view + ".", e);
            throw new RuntimeException(e);
        }
    }

    @Override
    public Map<String, T> getSingleViewFeatureMap(int view) {
        try {
            return selectFeatures(view);
        } catch (SQLException|IOException e) {
            logger.error("Could not get single view features for view " + view + ".", e);
            throw new RuntimeException(e);
        }
    }

    @Override
    public List<T> getFeatureVectors(String name) {
        try {
            return selectFeatures(name);
        } catch (SQLException|IOException e) {
            logger.error("Could not get all views features for name '" + name + "'.", e);
            throw new RuntimeException(e);
        }
    }

    @Override
    public Map<String, List<T>> getFeatureVectors(List<String> names) {
        try {
            return selectFeatures(names);
        } catch (SQLException|IOException e) {
            logger.error("Could not get all views features for the given list of names.", e);
            throw new RuntimeException(e);
        }
    }

    @Override
    public Map<String, List<T>> getFeatureMap() {
        try {
            return selectAllFeatures();
        } catch (SQLException|IOException e) {
            logger.error("Could not get all views features for all the names in the database.", e);
            throw new RuntimeException(e);
        }
    }

    @Override
    public boolean writeFeatureMapToStream(OutputStream outputStream) {
        throw new NotImplementedException();
    }

    protected void createDatabase() {
        try {
            Statement statement = connection.createStatement();
            statement.executeUpdate("DROP DATABASE IF EXISTS learn");
            statement.executeUpdate("CREATE DATABASE learn");
            String tableCreationQuery = "CREATE TABLE learn.features (name VARCHAR(100)";
            for (int view = 0; view < numberOfViews; view++)
                tableCreationQuery += ", features_view_" + view + " MEDIUMBLOB";
            tableCreationQuery += ", PRIMARY KEY (name) )";
            statement.executeUpdate(tableCreationQuery);
            statement.close();
        } catch (SQLException e) {
            logger.error("Could not create the database.");
        }
    }

    private void insertFeatures(String name, T features, int view) throws SQLException {
        String insertionQuery = "INSERT INTO learn.features (name, features_view_" + view + ") VALUES (?, ?) " +
                "ON DUPLICATE KEY UPDATE features_view_" + view + "=VALUES(features_view_" + view + ")";
        PreparedStatement preparedStatement = connection.prepareStatement(insertionQuery);
        preparedStatement.setString(1, name);
        if (features != null)
            preparedStatement.setBlob(2, features.getEncoder(true));
        else
            preparedStatement.setBlob(2, (InputStream) null);
        preparedStatement.executeUpdate();
        preparedStatement.close();
    }

    private void insertFeatures(Map<String, T> featureMappings, int view) throws SQLException {
        String insertionQuery = "INSERT INTO learn.features (name, features_view_" + view + ") VALUES ";
        for (int i = 0; i < featureMappings.size(); i++)
            insertionQuery += "(?, ?), ";
        insertionQuery = insertionQuery.substring(0, insertionQuery.length() - 2)
                + " ON DUPLICATE KEY UPDATE features_view_" + view + "=VALUES(features_view_" + view + ")";
        PreparedStatement preparedStatement = connection.prepareStatement(insertionQuery);
        int parameterIndex = 1;
        for (Map.Entry<String, T> features : featureMappings.entrySet()) {
            preparedStatement.setString(parameterIndex++, features.getKey());
            T featuresVector = features.getValue();
            if (featuresVector != null)
                preparedStatement.setBlob(parameterIndex++, featuresVector.getEncoder(true));
            else
                preparedStatement.setBlob(parameterIndex++, (InputStream) null);
        }
        preparedStatement.executeUpdate();
        preparedStatement.close();
    }

    private void insertFeatures(String name, List<T> features) throws SQLException {
        String insertionQuery = "INSERT INTO learn.features VALUES (?";
        String lastPartOfInsertionQuery = "ON DUPLICATE KEY UPDATE ";
        for (int view = 0; view < numberOfViews; view++) {
            insertionQuery += ", ?";
            if (view > 0)
                lastPartOfInsertionQuery += ", features_view_" + view + "=VALUES(features_view_" + view + ")";
            else
                lastPartOfInsertionQuery += "features_view_" + view + "=VALUES(features_view_" + view + ")";
        }
        insertionQuery += ") " + lastPartOfInsertionQuery;
        PreparedStatement preparedStatement = connection.prepareStatement(insertionQuery);
        preparedStatement.setString(1, name);
        for (int view = 0; view < numberOfViews; view++) {
            T featuresVector = features.get(view);
            if (featuresVector != null)
                preparedStatement.setBlob(view + 2, featuresVector.getEncoder(true));
            else
                preparedStatement.setBlob(view + 2, (InputStream) null);
        }
        preparedStatement.executeUpdate();
        preparedStatement.close();
    }

    private void insertFeatures(Map<String, List<T>> featureMappings) throws SQLException {
        String insertionQuery = "INSERT INTO learn.features VALUES ";
        String valuesPartOfInsertionQuery = "(?";
        String lastPartOfInsertionQuery = "ON DUPLICATE KEY UPDATE ";
        for (int view = 0; view < numberOfViews; view++) {
            valuesPartOfInsertionQuery += ", ?";
            if (view > 0)
                lastPartOfInsertionQuery += ", features_view_" + view + "=VALUES(features_view_" + view + ")";
            else
                lastPartOfInsertionQuery += "features_view_" + view + "=VALUES(features_view_" + view + ")";
        }
        valuesPartOfInsertionQuery += "), ";
        for (int i = 0; i < featureMappings.size(); i++)
            insertionQuery += valuesPartOfInsertionQuery;
        insertionQuery = insertionQuery.substring(0, insertionQuery.length() - 2) + lastPartOfInsertionQuery;
        PreparedStatement preparedStatement = connection.prepareStatement(insertionQuery);
        int parameterIndex = 1;
        for (Map.Entry<String, List<T>> features : featureMappings.entrySet()) {
            preparedStatement.setString(parameterIndex++, features.getKey());
            for (int view = 0; view < numberOfViews; view++) {
                T featuresVector = features.getValue().get(view);
                if (featuresVector != null)
                    preparedStatement.setBlob(parameterIndex++, featuresVector.getEncoder(true));
                else
                    preparedStatement.setBlob(parameterIndex++, (InputStream) null);
            }
        }
        preparedStatement.executeUpdate();
        preparedStatement.close();
    }

    @SuppressWarnings("unchecked")
    private T selectFeatures(String name, int view) throws SQLException, IOException {
        String selectionQuery = "SELECT features_view_" + view + " FROM learn.features WHERE name='" + name + "'";
        ResultSet result = connection.createStatement().executeQuery(selectionQuery);
        if (result.next()) {
            InputStream inputStream = result.getBinaryStream("features_view_" + view);
            if (inputStream != null)
                return (T) Vectors.build(inputStream);
            else
                return null;
        } else {
            return null;
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, T> selectFeatures(List<String> names, int view) throws SQLException, IOException {
        String selectionQuery = "SELECT name, features_view_" + view + " FROM learn.features WHERE name IN (";
        for (String name : names)
            selectionQuery += "'" + name + "', ";
        selectionQuery = selectionQuery.substring(0, selectionQuery.length() - 1) + ")";
        ResultSet result = connection.createStatement().executeQuery(selectionQuery);
        Map<String, T> features = new HashMap<>();
        while (result.next()) {
            InputStream inputStream = result.getBinaryStream("features_view_" + view);
            if (inputStream != null)
                features.put(result.getString("name"), (T) Vectors.build(inputStream));
            else
                features.put(result.getString("name"), null);
        }
        return features;
    }

    @SuppressWarnings("unchecked")
    private Map<String, T> selectFeatures(int view) throws SQLException, IOException {
        String selectionQuery = "SELECT name, features_view_" + view + " FROM learn.features";
        ResultSet result = connection.createStatement().executeQuery(selectionQuery);
        Map<String, T> features = new HashMap<>();
        while (result.next()) {
            InputStream inputStream = result.getBinaryStream("features_view_" + view);
            if (inputStream != null)
                features.put(result.getString("name"), (T) Vectors.build(inputStream));
            else
                features.put(result.getString("name"), null);
        }
        return features;
    }

    @SuppressWarnings("unchecked")
    private List<T> selectFeatures(String name) throws SQLException, IOException {
        String selectionQuery = "SELECT";
        for (int view = 0; view < numberOfViews; view++)
            selectionQuery += " features_view_" + view;
        selectionQuery += " FROM learn.features WHERE name='" + name + "'";
        ResultSet result = connection.createStatement().executeQuery(selectionQuery);
        if (result.next()) {
            List<T> features = new ArrayList<>();
            for (int view = 0; view < numberOfViews; view++) {
                InputStream inputStream = result.getBinaryStream("features_view_" + view);
                if (inputStream != null)
                    features.add((T) Vectors.build(inputStream));
                else
                    features.add(null);
            }
            return features;
        } else {
            return null;
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, List<T>> selectFeatures(List<String> names) throws SQLException, IOException {
        String selectionQuery = "SELECT * FROM learn.features WHERE name IN (";
        for (String name : names)
            selectionQuery += "'" + name + "', ";
        selectionQuery = selectionQuery.substring(0, selectionQuery.length() - 1) + ")";
        ResultSet result = connection.createStatement().executeQuery(selectionQuery);
        Map<String, List<T>> featureMappings = new HashMap<>();
        while (result.next()) {
            List<T> features = new ArrayList<>();
            for (int view = 0; view < numberOfViews; view++) {
                InputStream inputStream = result.getBinaryStream("features_view_" + view);
                if (inputStream != null)
                    features.add((T) Vectors.build(inputStream));
                else
                    features.add(null);
            }
            featureMappings.put(result.getString("name"), features);
        }
        return featureMappings;
    }

    @SuppressWarnings("unchecked")
    private Map<String, List<T>> selectAllFeatures() throws SQLException, IOException {
        String selectionQuery = "SELECT * FROM learn.features";
        ResultSet result = connection.createStatement().executeQuery(selectionQuery);
        Map<String, List<T>> features = new HashMap<>();
        while (result.next()) {
            List<T> featuresForName = new ArrayList<>();
            for (int view = 0; view < numberOfViews; view++) {
                InputStream inputStream = result.getBinaryStream("features_view_" + view);
                if (inputStream != null)
                    featuresForName.add((T) Vectors.build(inputStream));
                else
                    featuresForName.add(null);
            }
            features.put(result.getString("name"), featuresForName);
        }
        return features;
    }

    public static void main(String[] args) {
        FeatureMapMariaDB storage = new FeatureMapMariaDB(1, "jdbc:mariadb://localhost/", "root", null);
        storage.createDatabase();
        File inputFile = new File("/Volumes/Macintosh HD/Users/Anthony/Development/NELL/data/features/adjectives1.features");
        try {
            storage.loadFeatureMap(new FileInputStream(inputFile));
            storage.connection.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
//        try {
//            Connection connection = DriverManager.getConnection("jdbc:mariadb://localhost/test", "org.platanios", null);
//            Statement stmt = connection.createStatement();
//            stmt.executeUpdate("CREATE TABLE d (id int not null primary key, value varchar(20))");
//            stmt.close();
//            connection.close();
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
    }
}
