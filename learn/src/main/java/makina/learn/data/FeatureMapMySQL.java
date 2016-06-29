package makina.learn.data;

import makina.math.matrix.Vector;
import makina.math.matrix.Vectors;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.sql.*;
import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class FeatureMapMySQL<T extends Vector> extends FeatureMap<T> {
    private final String databaseName;
    private final String tableName;

    private final Connection connection;

    public FeatureMapMySQL(int numberOfViews, String host, String username, String password) {
        this(numberOfViews, host, username, password, "learn", "features");
    }

    public FeatureMapMySQL(int numberOfViews,
                           String host,
                           String username,
                           String password,
                           String databaseName,
                           String tableName) {
        super(numberOfViews);
        this.databaseName = databaseName;
        this.tableName = tableName;
        try {
            connection = DriverManager.getConnection(host, username, password);
        } catch (SQLException e) {
            logger.error("Could not connect to the default database!", e);
            throw new RuntimeException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public List<String> getNames() {
        try {
            return selectNames();
        } catch (SQLException e) {
            logger.error("Could not get the names from the database.", e);
            throw new RuntimeException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void addFeatureMappings(String name, T features, int view) {
        try {
            insertFeatures(name, features, view);
        } catch (SQLException e) {
            logger.error("Could not insert single view features for name '" + name + "' and view " + view + ".", e);
            throw new RuntimeException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void addFeatureMappings(Map<String, T> featureMappings, int view) {
        try {
            insertFeatures(featureMappings, view);
        } catch (SQLException e) {
            logger.error("Could not insert single view feature mappings for view " + view + ".", e);
            throw new RuntimeException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void addFeatureMappings(String name, List<T> features) {
        if (features.size() != numberOfViews) {
            logger.error("The size of the provided list must be equal to number of views.");
            throw new RuntimeException("The size of the provided list must be equal to number of views.");
        }
        try {
            insertFeatures(name, features);
        } catch (SQLException e) {
            logger.error("Could not insert multiple views features for name '" + name + "'.", e);
            throw new RuntimeException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void addFeatureMappings(Map<String, List<T>> featureMappings) {
        for (Map.Entry<String, List<T>> featureMapEntry : featureMappings.entrySet())
            if (featureMapEntry.getValue().size() != numberOfViews) {
                logger.error("All lists in the provided map must have size equal to number of views.");
                throw new RuntimeException("All lists in the provided map must have size equal to number of views.");
            }
        try {
            insertFeatures(featureMappings);
        } catch (SQLException e) {
            logger.error("Could not insert multiple views features for multiple names.", e);
            throw new RuntimeException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public T getFeatureVector(String name, int view) {
        try {
            return selectFeatures(name, view);
        } catch (SQLException|IOException e) {
            logger.error("Could not get single view features for name '" + name + "' and view " + view + ".", e);
            throw new RuntimeException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Map<String, T> getFeatureVectors(List<String> names, int view) {
        try {
            return selectFeatures(names, view);
        } catch (SQLException|IOException e) {
            logger.error("Could not get single view features for the given list of names and view " + view + ".", e);
            throw new RuntimeException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Map<String, T> getFeatureMap(int view) {
        try {
            return selectFeatures(view);
        } catch (SQLException|IOException e) {
            logger.error("Could not get single view features for view " + view + ".", e);
            throw new RuntimeException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public List<T> getFeatureVectors(String name) {
        try {
            return selectFeatures(name);
        } catch (SQLException|IOException e) {
            logger.error("Could not get all views features for name '" + name + "'.", e);
            throw new RuntimeException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Map<String, List<T>> getFeatureVectors(List<String> names) {
        try {
            return selectFeatures(names);
        } catch (SQLException|IOException e) {
            logger.error("Could not get all views features for the given list of names.", e);
            throw new RuntimeException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Map<String, List<T>> getFeatureMap() {
        try {
            return selectAllFeatures();
        } catch (SQLException|IOException e) {
            logger.error("Could not get all views features for all the names in the database.", e);
            throw new RuntimeException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean write(OutputStream outputStream) {
        throw new UnsupportedOperationException();
    }

    public void createDatabase() {
        try {
            Statement statement = connection.createStatement();
            statement.executeUpdate("DROP DATABASE IF EXISTS " + databaseName);
            statement.executeUpdate("CREATE DATABASE " + databaseName);
            statement.close();
            createTables();
        } catch (SQLException e) {
            logger.error("Could not create the database.");
        }
    }

    public void createTables() {
        try {
            Statement statement = connection.createStatement();
            String tableCreationQuery = "CREATE TABLE " + databaseName + "." + tableName + " (name VARCHAR(100)";
            for (int view = 0; view < numberOfViews; view++)
                tableCreationQuery += ", features_view_" + view + " MEDIUMBLOB";
            tableCreationQuery += ", PRIMARY KEY (name) )";
            statement.executeUpdate(tableCreationQuery);
            statement.close();
        } catch (SQLException e) {
            logger.error("Could not create the database tables.");
        }
    }

    private void insertFeatures(String name, T features, int view) throws SQLException {
        String insertionQuery = "INSERT INTO " + databaseName + "." + tableName +
                " (name, features_view_" + view + ") VALUES (?, ?) " + "ON DUPLICATE KEY UPDATE features_view_" +
                view + "=VALUES(features_view_" + view + ")";
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
        String insertionQuery = "INSERT INTO " + databaseName + "." + tableName + " (name, features_view_" +
                view + ") VALUES ";
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
        String insertionQuery = "INSERT INTO " + databaseName + "." + tableName + " VALUES (?";
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
        String insertionQuery = "INSERT INTO " + databaseName + "." + tableName + " VALUES ";
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

    private List<String> selectNames() throws SQLException {
        List<String> names = new ArrayList<>();
        String selectionQuery = "SELECT name FROM " + databaseName + "." + tableName;
        ResultSet result = connection.createStatement().executeQuery(selectionQuery);
        while (result.next())
            names.add(result.getString("name"));
        return names;
    }

    @SuppressWarnings("unchecked")
    private T selectFeatures(String name, int view) throws SQLException, IOException {
        String selectionQuery = "SELECT features_view_" + view + " FROM " + databaseName + "." + tableName +
                " WHERE name=?";
        PreparedStatement preparedStatement = connection.prepareStatement(selectionQuery);
        preparedStatement.setString(1, name);
        ResultSet result = preparedStatement.executeQuery();
        if (result.next()) {
            InputStream inputStream = result.getBinaryStream("features_view_" + view);
            if (inputStream != null) {
                preparedStatement.close();
                return (T) Vectors.build(inputStream);
            } else {
                preparedStatement.close();
                return null;
            }
        } else {
            preparedStatement.close();
            return null;
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, T> selectFeatures(List<String> names, int view) throws SQLException, IOException {
        String selectionQuery = "SELECT name, features_view_" + view + " FROM " + databaseName + "." + tableName +
                " WHERE name IN (";
        for (String name : names)
            selectionQuery += "?, ";
        selectionQuery = selectionQuery.substring(0, selectionQuery.length() - 1) + ")";
        PreparedStatement preparedStatement = connection.prepareStatement(selectionQuery);
        int parameterIndex = 1;
        for (String name : names)
            preparedStatement.setString(parameterIndex++, name);
        ResultSet result = preparedStatement.executeQuery();
        Map<String, T> features = new HashMap<>();
        while (result.next()) {
            InputStream inputStream = result.getBinaryStream("features_view_" + view);
            if (inputStream != null)
                features.put(result.getString("name"), (T) Vectors.build(inputStream));
            else
                features.put(result.getString("name"), null);
        }
        preparedStatement.close();
        return features;
    }

    @SuppressWarnings("unchecked")
    private Map<String, T> selectFeatures(int view) throws SQLException, IOException {
        String selectionQuery = "SELECT name, features_view_" + view + " FROM " + databaseName + "." + tableName;
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
            selectionQuery += " features_view_" + view + ",";
        selectionQuery = selectionQuery.substring(0, selectionQuery.length() - 1) + " FROM " + databaseName + "." + tableName + " WHERE name=?";
        PreparedStatement preparedStatement = connection.prepareStatement(selectionQuery);
        preparedStatement.setString(1, name);
        ResultSet result = preparedStatement.executeQuery();
        if (result.next()) {
            List<T> features = new ArrayList<>();
            for (int view = 0; view < numberOfViews; view++) {
                InputStream inputStream = result.getBinaryStream("features_view_" + view);
                if (inputStream != null)
                    features.add((T) Vectors.build(inputStream));
                else
                    features.add(null);
            }
            preparedStatement.close();
            return features;
        } else {
            preparedStatement.close();
            List<T> features = new ArrayList<>();
            for (int view = 0; view < numberOfViews; view++)
                features.add(null);
            return features;
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, List<T>> selectFeatures(List<String> names) throws SQLException, IOException {
        String selectionQuery = "SELECT * FROM " + databaseName + "." + tableName + " WHERE name IN (";
        for (String name : names)
            selectionQuery += "?, ";
        selectionQuery = selectionQuery.substring(0, selectionQuery.length() - 1) + ")";
        PreparedStatement preparedStatement = connection.prepareStatement(selectionQuery);
        int parameterIndex = 1;
        for (String name : names)
            preparedStatement.setString(parameterIndex++, name);
        ResultSet result = preparedStatement.executeQuery();
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
        preparedStatement.close();
        return featureMappings;
    }

    @SuppressWarnings("unchecked")
    private Map<String, List<T>> selectAllFeatures() throws SQLException, IOException {
        String selectionQuery = "SELECT * FROM " + databaseName + "." + tableName;
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
}
