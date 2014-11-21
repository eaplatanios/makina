package org.platanios.learn.classification;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.math.matrix.Vector;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.*;
import java.sql.*;
import java.util.ArrayList;
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
            logger.error("Could not connect to the default database server!");
            logger.catching(e);
            throw new RuntimeException("Could not connect to the default database server.");
        }
    }

    protected FeatureMapMariaDB(int numberOfViews, String host, String username, String password) {
        super(numberOfViews);

        try {
            connection = DriverManager.getConnection(host, username, password);
        } catch (SQLException e) {
            logger.error("Could not connect to the default database!");
            logger.catching(e);
            throw new RuntimeException("Could not connect to the database server.");
        }
    }

    public void createDatabase() {
        try {
            Statement statement = connection.createStatement();
            statement.executeUpdate("DROP DATABASE IF EXISTS learn");
            statement.executeUpdate("CREATE DATABASE learn");
            String tableCreationQuery = "CREATE TABLE learn.features ( input VARCHAR(100)";
            for (int view = 0; view < numberOfViews; view++)
                tableCreationQuery += ", features_view_" + view + " MEDIUMBLOB";
            tableCreationQuery += ", PRIMARY KEY (input) )";
            statement.executeUpdate(tableCreationQuery);
            statement.close();
        } catch (SQLException e) {
            logger.error("Could not create the database.");
        }
    }

    @Override
    @SuppressWarnings("unchecked")
    public void loadFeatureMap(ObjectInputStream inputStream) {
        try {
            if (numberOfViews != inputStream.readInt()) {
                logger.error("This feature map was initialized for a number of views that is different than the " +
                                     "number of feature views stored in the input stream.");
                throw new RuntimeException("This feature map was initialized for a number of views that is different " +
                                                   "than the number of feature views stored in the input stream.");
            }
            int numberOfKeys = inputStream.readInt();
            for (int i = 0; i < numberOfKeys; i++) {
                String name = (String) inputStream.readObject();
                List<T> features = new ArrayList<>();
                for (int view = 0; view < numberOfViews; view++)
                    features.add((T) inputStream.readObject());
                insertFeatures(name, features);
            }
            inputStream.close();
            logger.debug("Loaded the feature map from an input stream.");
        } catch (Exception e) {
            logger.error("An exception was thrown while loading the feature map from an input stream.", e);
        }
    }

    @Override
    public void loadFeatureMap(Connection databaseConnection) {
        throw new NotImplementedException();
    }

    private void insertFeatures(String name, List<T> features) throws SQLException, IOException {
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
            ByteArrayOutputStream byteOutputStream = new ByteArrayOutputStream();
            ObjectOutputStream outputStream = new ObjectOutputStream(byteOutputStream);
            outputStream.writeObject(features.get(view));
            outputStream.close();
            preparedStatement.setBytes(view + 2, byteOutputStream.toByteArray());
        }
        preparedStatement.executeUpdate();
        preparedStatement.close();
    }

    private void insertFeature(String name, T features, int view) throws SQLException, IOException {
        String insertionQuery = "INSERT INTO learn.features (input, features_view_" + view + ") VALUES (?, ?) " +
                "ON DUPLICATE KEY UPDATE features_view_" + view + "=VALUES(features_view_" + view + ")";
        PreparedStatement preparedStatement = connection.prepareStatement(insertionQuery);
        preparedStatement.setString(1, name);
        ByteArrayOutputStream byteOutputStream = new ByteArrayOutputStream();
        ObjectOutputStream outputStream = new ObjectOutputStream(byteOutputStream);
        outputStream.writeObject(features);
        outputStream.close();
        preparedStatement.setBytes(2, byteOutputStream.toByteArray());
        preparedStatement.executeUpdate();
        preparedStatement.close();
    }

    @Override
    public void addSingleViewFeatureMappings(String name, T features, int view) {
        throw new NotImplementedException();
    }

    @Override
    public void addSingleViewFeatureMappings(Map<String, T> featureMappings, int view) {
        throw new NotImplementedException();
    }

    @Override
    public void addFeatureMappings(String name, List<T> features) {
        throw new NotImplementedException();
    }

    @Override
    public void addFeatureMappings(Map<String, List<T>> featureMappings) {
        throw new NotImplementedException();
    }

    @Override
    public T getSingleViewFeatureVector(String name, int view) {
        throw new NotImplementedException();
    }

    @Override
    public Map<String, T> getSingleViewFeatureVectors(List<String> names, int view) {
        throw new NotImplementedException();
    }

    @Override
    public Map<String, T> getSingleViewFeatureMap(int view) {
        throw new NotImplementedException();
    }

    @Override
    public List<T> getFeatureVectors(String name) {
        throw new NotImplementedException();
    }

    @Override
    public Map<String, List<T>> getFeatureVectors(List<String> names) {
        throw new NotImplementedException();
    }

    @Override
    public Map<String, List<T>> getFeatureMap() {
        throw new NotImplementedException();
    }

    @Override
    public boolean writeFeatureMapToStream(ObjectOutputStream outputStream) {
        throw new NotImplementedException();
    }

    public static void main(String[] args) {
        FeatureMapMariaDB storage = new FeatureMapMariaDB(1, "jdbc:mariadb://localhost/", "root", null);
        storage.createDatabase();
        File inputFile = new File("/Volumes/Macintosh HD/Users/Anthony/Development/NELL/data/features/adjectives1.features");
        try {
            storage.loadFeatureMap(new ObjectInputStream(new FileInputStream(inputFile)));
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
