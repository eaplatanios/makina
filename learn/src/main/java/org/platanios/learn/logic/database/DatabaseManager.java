package org.platanios.learn.logic.database;

import org.hibernate.*;
import org.hibernate.boot.registry.StandardServiceRegistryBuilder;
import org.hibernate.cfg.Configuration;
import org.hibernate.criterion.Projections;
import org.hibernate.criterion.Restrictions;
import org.hibernate.service.ServiceRegistry;
import org.platanios.learn.logic.Logic;
import org.platanios.learn.logic.formula.EntityType;
import org.platanios.learn.logic.formula.Predicate;
import org.platanios.learn.logic.grounding.GroundPredicate;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * TODO: Change the code so that we do not keep opening and closing sessions, but rather use the getCurrentSession() method.
 *
 * @author Emmanouil Antonios Platanios
 */
public class DatabaseManager {
    // Connection Settings
    private final String driverClass;
    private final String connectionURL;
    private final String username;
    private final String password;
    private final int connectionPoolSize;
    private final String dialect;

    // Other Settings
    private final String currentSessionContextClass;
    private final boolean showSQL;
    private final boolean newGeneratorMappings;
    private final int jdbcBatchSize;
    private final String hbm2ddlAuto;

    private SessionFactory sessionFactory;

    public static class Builder {
        private String driverClass = "org.mariadb.jdbc.Driver";
        private String connectionURL = "jdbc:mariadb://localhost/learn_logic";
        private String username = "root";
        private String password = "";
        private int connectionPoolSize = 1;
        private String dialect = "org.hibernate.dialect.MySQLDialect";
        private String currentSessionContextClass = "thread";
        private boolean showSQL = false;
        private boolean newGeneratorMappings = true;
        private int jdbcBatchSize = 20;
        private String hbm2ddlAuto = "create-drop";

        public Builder() {

        }

        public Builder driverClass(String driverClass) {
            this.driverClass = driverClass;
            return this;
        }

        public Builder connectionURL(String connectionURL) {
            this.connectionURL = connectionURL;
            return this;
        }

        public Builder username(String username) {
            this.username = username;
            return this;
        }

        public Builder password(String password) {
            this.password = password;
            return this;
        }

        public Builder connectionPoolSize(int connectionPoolSize) {
            this.connectionPoolSize = connectionPoolSize;
            return this;
        }

        public Builder dialect(String dialect) {
            this.dialect = dialect;
            return this;
        }

        public Builder currentSessionContextClass(String currentSessionContextClass) {
            this.currentSessionContextClass = currentSessionContextClass;
            return this;
        }

        public Builder showSQL(boolean showSQL) {
            this.showSQL = showSQL;
            return this;
        }

        public Builder newGeneratorMappings(boolean newGeneratorMappings) {
            this.newGeneratorMappings = newGeneratorMappings;
            return this;
        }

        public Builder jdbcBatchSize(int jdbcBatchSize) {
            this.jdbcBatchSize = jdbcBatchSize;
            return this;
        }

        public Builder hbm2ddlAuto(String hbm2ddlAuto) {
            this.hbm2ddlAuto = hbm2ddlAuto;
            return this;
        }

        public DatabaseManager build() {
            return new DatabaseManager(this);
        }
    }

    private DatabaseManager(Builder builder) {
        driverClass = builder.driverClass;
        connectionURL = builder.connectionURL;
        username = builder.username;
        password = builder.password;
        connectionPoolSize = builder.connectionPoolSize;
        dialect = builder.dialect;
        currentSessionContextClass = builder.currentSessionContextClass;
        showSQL = builder.showSQL;
        newGeneratorMappings = builder.newGeneratorMappings;
        jdbcBatchSize = builder.jdbcBatchSize;
        hbm2ddlAuto = builder.hbm2ddlAuto;

        buildSessionFactory();
    }

    public void buildSessionFactory() {
        Configuration configuration =
                new Configuration()
                        .setProperty("hibernate.connection.driver_class", driverClass)
                        .setProperty("hibernate.connection.url", connectionURL)
                        .setProperty("hibernate.connection.username", username)
                        .setProperty("hibernate.connection.password", password)
                        .setProperty("hibernate.connection.pool_size", Integer.toString(connectionPoolSize))
                        .setProperty("hibernate.dialect", dialect)
                        .setProperty("hibernate.current_session_context_class", currentSessionContextClass)
                        .setProperty("hibernate.show_sql", Boolean.toString(showSQL))
                        .setProperty("hibernate.id.new_generator_mappings", Boolean.toString(newGeneratorMappings))
                        .setProperty("hibernate.jdbc.batch_size", Integer.toString(jdbcBatchSize))
                        .setProperty("hibernate.hbm2ddl.auto", hbm2ddlAuto)
                        .addAnnotatedClass(DatabaseEntityType.class)
                        .addAnnotatedClass(DatabaseEntityTypeValue.class)
                        .addAnnotatedClass(DatabasePredicate.class)
                        .addAnnotatedClass(DatabasePredicateArgumentType.class)
                        .addAnnotatedClass(DatabaseGroundPredicate.class)
                        .addAnnotatedClass(DatabaseGroundPredicateArgument.class)
                        .addAnnotatedClass(DatabaseGroundFormula.class)
                        .addAnnotatedClass(DatabaseGroundFormulaPredicate.class);
        ServiceRegistry serviceRegistry = new StandardServiceRegistryBuilder()
                .applySettings(configuration.getProperties())
                .build();
        sessionFactory = configuration.buildSessionFactory(serviceRegistry);
    }

    public void closeSessionFactory() {
        sessionFactory.close();
    }

    private void insertObject(Object object) {
        Session session = sessionFactory.openSession();
        Transaction transaction = session.beginTransaction();
        session.save(object);
        transaction.commit();
        session.close();
    }

    private void insertObjects(List<?> objects) {
        Session session = sessionFactory.openSession();
        Transaction transaction = session.beginTransaction();
        int numberOfObjectsSaved = 0;
        for (Object object : objects) {
            session.save(object);
            if (++numberOfObjectsSaved % jdbcBatchSize == 0) {
                session.flush();
                session.clear();
            }
        }
        transaction.commit();
        session.close();
    }

    private Object getObject(Class<?> type, long id) {
        Session session = sessionFactory.openSession();
        Object object = session.get(type, id);
        session.close();
        return object;
    }

    private Object getObjectProxy(Class<?> type, long id) {
        Session session = sessionFactory.openSession();
        Object object = session.load(type, id);
        session.close();
        return object;
    }

    private boolean checkIfObjectExists(Class<?> type, long id) {
        return getObject(type, id) != null;
    }

    private long getNumberOfRows(Class<?> type) {
        Session session = sessionFactory.openSession();
        long numberOfRows = (long) session.createCriteria(type)
                .setProjection(Projections.rowCount())
                .uniqueResult();
        session.close();
        return numberOfRows;
    }

    public DatabaseEntityType addEntityType(List<Long> allowedValues) {
        DatabaseEntityType databaseEntityType = new DatabaseEntityType();
        databaseEntityType.setAllowedValues(
                allowedValues.stream()
                        .map(value -> new DatabaseEntityTypeValue(databaseEntityType, value))
                        .collect(Collectors.toList())
        );
        insertObject(databaseEntityType);
        return databaseEntityType;
    }

    public DatabaseEntityType addEntityType(String name, List<Long> allowedValues) {
        DatabaseEntityType databaseEntityType = new DatabaseEntityType(name);
        databaseEntityType.setAllowedValues(
                allowedValues.stream()
                        .map(value -> new DatabaseEntityTypeValue(databaseEntityType, value))
                        .collect(Collectors.toList())
        );
        insertObject(databaseEntityType);
        return databaseEntityType;
    }

    public EntityType getEntityType(long id) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(DatabaseEntityType.class);
        criteria.setFetchMode("allowedValues", FetchMode.JOIN);
        criteria.add(Restrictions.eq("id", id));
        DatabaseEntityType entityType = (DatabaseEntityType) criteria.uniqueResult();
        session.close();
        return convertToEntityType(entityType);
    }

    public EntityType getEntityType(String name) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(DatabaseEntityType.class);
        criteria.setFetchMode("allowedValues", FetchMode.JOIN);
        criteria.add(Restrictions.eq("name", name));
        DatabaseEntityType entityType = (DatabaseEntityType) criteria.uniqueResult();
        session.close();
        return convertToEntityType(entityType);
    }

    public DatabaseEntityType getDatabaseEntityType(long id) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(DatabaseEntityType.class);
        criteria.setFetchMode("allowedValues", FetchMode.JOIN);
        criteria.add(Restrictions.eq("id", id));
        DatabaseEntityType entityType = (DatabaseEntityType) criteria.uniqueResult();
        session.close();
        return entityType;
    }

    public DatabaseEntityType getDatabaseEntityType(String name) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(DatabaseEntityType.class);
        criteria.setFetchMode("allowedValues", FetchMode.JOIN);
        criteria.add(Restrictions.eq("name", name));
        DatabaseEntityType entityType = (DatabaseEntityType) criteria.uniqueResult();
        session.close();
        return entityType;
    }

    public List<DatabaseEntityTypeValue> getEntityTypeAllowedValues(long id) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(DatabaseEntityType.class);
        criteria.setFetchMode("allowedValues", FetchMode.JOIN);
        criteria.add(Restrictions.eq("id", id));
        DatabaseEntityType entityType = (DatabaseEntityType) criteria.uniqueResult();
        session.close();
        return entityType.getAllowedValues();
    }

    public DatabasePredicate addPredicate(boolean closed) {
        return addPredicate((String) null, closed);
    }

    public DatabasePredicate addPredicate(String name, boolean closed) {
        DatabasePredicate databasePredicate = new DatabasePredicate();
        databasePredicate.setName(name);
        databasePredicate.setClosed(closed);
        insertObject(databasePredicate);
        return databasePredicate;
    }

    public DatabasePredicate addPredicate(List<EntityType> argumentTypes,
                                          boolean closed) {
        return addPredicate(null, argumentTypes, closed);
    }

    public DatabasePredicate addPredicate(String name,
                                          List<EntityType> argumentTypes,
                                          boolean closed) {
        DatabasePredicate databasePredicate = new DatabasePredicate();
        databasePredicate.setName(name);
        databasePredicate.setClosed(closed);
        List<DatabasePredicateArgumentType> databasePredicateArgumentTypes = new ArrayList<>();
        for (int argumentIndex = 0; argumentIndex < argumentTypes.size(); argumentIndex++)
            databasePredicateArgumentTypes.add(
                    new DatabasePredicateArgumentType(databasePredicate,
                                                      argumentIndex,
                                                      getDatabaseEntityType(argumentTypes.get(argumentIndex).getId()))
            );
        databasePredicate.setArgumentTypes(databasePredicateArgumentTypes);
        insertObject(databasePredicate);
        return databasePredicate;
    }

    public Predicate getPredicate(long id) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(DatabasePredicate.class);
        criteria.setFetchMode("argumentTypes", FetchMode.JOIN);
        criteria.add(Restrictions.eq("id", id));
        Predicate predicate = convertToPredicate((DatabasePredicate) criteria.uniqueResult());
        session.close();
        return predicate;
    }

    public Predicate getPredicate(String name) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(DatabasePredicate.class);
        criteria.setFetchMode("argumentTypes", FetchMode.JOIN);
        criteria.add(Restrictions.eq("name", name));
        Predicate predicate = convertToPredicate((DatabasePredicate) criteria.uniqueResult());
        session.close();
        return predicate;
    }

    public DatabasePredicate getDatabasePredicate(long id) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(DatabasePredicate.class);
        criteria.setFetchMode("argumentTypes", FetchMode.JOIN);
        criteria.add(Restrictions.eq("id", id));
        DatabasePredicate predicate = (DatabasePredicate) criteria.uniqueResult();
        session.close();
        return predicate;
    }

    public DatabasePredicate getDatabasePredicate(String name) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(DatabasePredicate.class);
        criteria.setFetchMode("argumentTypes", FetchMode.JOIN);
        criteria.add(Restrictions.eq("name", name));
        DatabasePredicate predicate = (DatabasePredicate) criteria.uniqueResult();
        session.close();
        return predicate;
    }

    @SuppressWarnings("unchecked")
    public List<Predicate> getClosedPredicates() {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(DatabasePredicate.class);
        criteria.add(Restrictions.eq("closed", true));
        List<Predicate> closedPredicates = ((List<DatabasePredicate>) criteria.list())
                .stream()
                .map(this::convertToPredicate)
                .collect(Collectors.toList());
        session.close();
        return closedPredicates;
    }

    public DatabaseGroundPredicate addGroundPredicate(long predicateId,
                                                      List<Long> variablesAssignment,
                                                      String value,
                                                      Class valueClass) {
        DatabasePredicate databasePredicate = (DatabasePredicate) getObject(DatabasePredicate.class, predicateId);
        DatabaseGroundPredicate databaseGroundPredicate = new DatabaseGroundPredicate(databasePredicate,
                                                                                      value,
                                                                                      valueClass);
        List<DatabaseGroundPredicateArgument> databaseGroundPredicateArguments = new ArrayList<>();
        for (int argumentIndex = 0; argumentIndex < variablesAssignment.size(); argumentIndex++)
            databaseGroundPredicateArguments.add(new DatabaseGroundPredicateArgument(
                    databaseGroundPredicate,
                    argumentIndex,
                    variablesAssignment.get(argumentIndex)
            ));
        databaseGroundPredicate.setGroundPredicateArguments(databaseGroundPredicateArguments);
        insertObject(databaseGroundPredicate);
        return databaseGroundPredicate;
    }

    @SuppressWarnings("unchecked")
    public boolean checkIfGroundPredicateExists(long predicateId, List<Long> variablesAssignment) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(DatabaseGroundPredicate.class);
        criteria.setFetchMode("groundPredicateArguments", FetchMode.JOIN);
        Criteria joinCriteria = criteria.createCriteria("predicate");
        joinCriteria.add(Restrictions.eq("id", predicateId));
        List<DatabaseGroundPredicate> databaseGroundPredicates = criteria.list();
        session.close();
        for (DatabaseGroundPredicate databaseGroundPredicate : databaseGroundPredicates) {
            List<DatabaseGroundPredicateArgument> databaseGroundPredicateArguments =
                    databaseGroundPredicate.getGroundPredicateArguments();
            boolean groundPredicateFound = true;
            for (DatabaseGroundPredicateArgument databaseGroundPredicateArgument : databaseGroundPredicateArguments)
                if (databaseGroundPredicateArgument.getArgumentValue()
                        != variablesAssignment.get(databaseGroundPredicateArgument.getArgumentIndex()))
                    groundPredicateFound = false;
            if (groundPredicateFound)
                return true;
        }
        return false;
    }

    public long getNumberOfGroundPredicates() {
        return getNumberOfRows(DatabaseGroundPredicate.class);
    }

    @SuppressWarnings("unchecked")
    public <R> List<GroundPredicate<R>> getGroundPredicates() {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(DatabaseGroundPredicate.class);
        List<GroundPredicate<R>> groundPredicates = ((List<DatabaseGroundPredicate>) criteria.list())
                .stream()
                .map(this::<R>convertToGroundPredicate)
                .collect(Collectors.toList());
        session.close();
        return groundPredicates;
    }

    @SuppressWarnings("unchecked")
    public <R> GroundPredicate<R> getGroundPredicate(long id) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(DatabaseGroundPredicate.class);
        criteria.setFetchMode("groundPredicateArguments", FetchMode.JOIN);
        criteria.add(Restrictions.eq("id", id));
        GroundPredicate<R> groundPredicate =
                convertToGroundPredicate((DatabaseGroundPredicate) criteria.uniqueResult());
        session.close();
        return groundPredicate;
    }

    @SuppressWarnings("unchecked")
    public <R> GroundPredicate<R> getGroundPredicate(long predicateId, List<Long> variablesAssignment) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(DatabaseGroundPredicate.class);
        criteria.setFetchMode("groundPredicateArguments", FetchMode.JOIN);
        Criteria joinCriteria = criteria.createCriteria("predicate");
        joinCriteria.add(Restrictions.eq("id", predicateId));
        List<DatabaseGroundPredicate> databaseGroundPredicates = criteria.list();
        for (DatabaseGroundPredicate databaseGroundPredicate : databaseGroundPredicates) {
            List<DatabaseGroundPredicateArgument> databaseGroundPredicateArguments =
                    databaseGroundPredicate.getGroundPredicateArguments();
            boolean groundPredicateFound = true;
            for (DatabaseGroundPredicateArgument databaseGroundPredicateArgument : databaseGroundPredicateArguments)
                if (databaseGroundPredicateArgument.getArgumentValue()
                        != variablesAssignment.get(databaseGroundPredicateArgument.getArgumentIndex()))
                    groundPredicateFound = false;
            if (groundPredicateFound) {
                GroundPredicate<R> groundPredicate = convertToGroundPredicate(databaseGroundPredicate);
                session.close();
                return groundPredicate;
            }
        }
        session.close();
        return null;
    }

    @SuppressWarnings("unchecked")
    public <R> R getPredicateAssignmentTruthValue(Predicate predicate, List<Long> variablesAssignment, Logic<R> logic) {
        Session session = sessionFactory.openSession();
        Criteria criteria = session.createCriteria(DatabaseGroundPredicate.class);
        criteria.setFetchMode("groundPredicateArguments", FetchMode.JOIN);
        Criteria joinCriteria = criteria.createCriteria("predicate");
        joinCriteria.add(Restrictions.eq("id", predicate.getId()));
        List<DatabaseGroundPredicate> databaseGroundPredicates = criteria.list();
        for (DatabaseGroundPredicate databaseGroundPredicate : databaseGroundPredicates) {
            List<DatabaseGroundPredicateArgument> databaseGroundPredicateArguments =
                    databaseGroundPredicate.getGroundPredicateArguments();
            boolean groundPredicateFound = true;
            for (DatabaseGroundPredicateArgument databaseGroundPredicateArgument : databaseGroundPredicateArguments)
                if (databaseGroundPredicateArgument.getArgumentValue()
                        != variablesAssignment.get(databaseGroundPredicateArgument.getArgumentIndex()))
                    groundPredicateFound = false;
            if (groundPredicateFound) {
                R truthValue = getGroundPredicateValue(databaseGroundPredicate);
                session.close();
                return truthValue;
            }
        }
        if (!getDatabasePredicate(predicate.getId()).getClosed()) {
            session.close();
            return null;
        } else {
            session.close();
            return logic.falseValue();
        }
    }

    public long getNumberOfEntityTypes() {
        return getNumberOfRows(DatabaseEntityType.class);
    }

    public boolean checkIfVariableTypeIDExists(long id) {
        return checkIfObjectExists(DatabaseEntityType.class, id);
    }

    private EntityType convertToEntityType(DatabaseEntityType databaseEntityType) {
        return new EntityType(databaseEntityType.getId(),
                              databaseEntityType.getName(),
                              databaseEntityType.getAllowedValues()
                                      .stream()
                                      .map(DatabaseEntityTypeValue::getValue)
                                      .collect(Collectors.toList()));
    }

    private Predicate convertToPredicate(DatabasePredicate databasePredicate) {
        return new Predicate(databasePredicate.getId(),
                             databasePredicate.getName(),
                             databasePredicate.getArgumentTypes()
                                     .stream()
                                     .map(DatabasePredicateArgumentType::getArgumentType)
                                     .map(databaseEntityType ->
                                                  new EntityType(databaseEntityType.getId(),
                                                                 databaseEntityType.getName(),
                                                                 databaseEntityType.getAllowedValues()
                                                                         .stream()
                                                                         .map(DatabaseEntityTypeValue::getValue)
                                                                         .collect(Collectors.toList())))
                                     .collect(Collectors.toList()));
    }

    @SuppressWarnings("unchecked")
    private <R> GroundPredicate<R> convertToGroundPredicate(DatabaseGroundPredicate databaseGroundPredicate) {
        List<DatabaseGroundPredicateArgument> databaseGroundPredicateArguments =
                databaseGroundPredicate.getGroundPredicateArguments();
        List<Long> argumentsAssignment = Arrays.asList(new Long[databaseGroundPredicateArguments.size()]);
        for (DatabaseGroundPredicateArgument databaseGroundPredicateArgument : databaseGroundPredicateArguments)
            argumentsAssignment.set(databaseGroundPredicateArgument.getArgumentIndex(),
                                    databaseGroundPredicateArgument.getArgumentValue());
        return new GroundPredicate<>(
                databaseGroundPredicate.getId(),
                new Predicate(databaseGroundPredicate.getPredicate().getId(),
                              databaseGroundPredicate.getPredicate().getName(),
                              databaseGroundPredicate.getPredicate().getArgumentTypes()
                                      .stream()
                                      .map(DatabasePredicateArgumentType::getArgumentType)
                                      .map(databaseEntityType ->
                                                   new EntityType(databaseEntityType.getId(),
                                                                  databaseEntityType.getName(),
                                                                  databaseEntityType.getAllowedValues()
                                                                          .stream()
                                                                          .map(DatabaseEntityTypeValue::getValue)
                                                                          .collect(Collectors.toList())))
                                      .collect(Collectors.toList())),
                argumentsAssignment,
                getGroundPredicateValue(databaseGroundPredicate)
        );
    }

    @SuppressWarnings("unchecked")
    private <R> R getGroundPredicateValue(DatabaseGroundPredicate databaseGroundPredicate) {
        R groundPredicateValue = null;
        if (databaseGroundPredicate.getValue() != null)
            if (databaseGroundPredicate.getValueClass() == Boolean.class)
                groundPredicateValue = (R) (Object) Boolean.parseBoolean(databaseGroundPredicate.getValue());
            else if (databaseGroundPredicate.getValueClass() == Integer.class)
                groundPredicateValue = (R) (Object) Integer.parseInt(databaseGroundPredicate.getValue());
            else if (databaseGroundPredicate.getValueClass() == Long.class)
                groundPredicateValue = (R) (Object) Long.parseLong(databaseGroundPredicate.getValue());
            else if (databaseGroundPredicate.getValueClass() == Float.class)
                groundPredicateValue = (R) (Object) Float.parseFloat(databaseGroundPredicate.getValue());
            else if (databaseGroundPredicate.getValueClass() == Double.class)
                groundPredicateValue = (R) (Object) Double.parseDouble(databaseGroundPredicate.getValue());
            else
                throw new IllegalStateException("Unsupported logic value type encountered.");
        return groundPredicateValue;
    }
}
