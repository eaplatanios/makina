package org.platanios.trade.data;

import org.hibernate.Criteria;
import org.hibernate.Session;
import org.hibernate.Transaction;
import org.hibernate.criterion.Property;
import org.hibernate.criterion.Restrictions;
import org.hibernate.sql.JoinType;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataManager {
    public static void addExchange(Exchange exchange) {
        insertObject(exchange);
    }

    public static void addExchanges(List<Exchange> exchanges) {
        insertObjects(exchanges);
    }

    public static void addDataVendor(DataVendor dataVendor) {
        insertObject(dataVendor);
    }

    public static void addDataVendors(List<DataVendor> dataVendors) {
        insertObjects(dataVendors);
    }

    public static void addSector(Sector sector) {
        insertObject(sector);
    }

    public static void addSectors(List<Sector> sectors) {
        insertObjects(sectors);
    }

    public static void addIndustryGroup(IndustryGroup industryGroup) {
        insertObject(industryGroup);
    }

    public static void addIndustryGroups(List<IndustryGroup> industryGroups) {
        insertObjects(industryGroups);
    }

    public static void addIndustry(Industry industry) {
        insertObject(industry);
    }

    public static void addIndustries(List<Industry> industries) {
        insertObjects(industries);
    }

    public static void addSubIndustry(SubIndustry subIndustry) {
        insertObject(subIndustry);
    }

    public static void addSubIndustries(List<SubIndustry> subIndustries) {
        insertObjects(subIndustries);
    }

    public static void addStock(Stock stock) {
        insertObject(stock);
    }

    public static void addStocks(List<Stock> stocks) {
        insertObjects(stocks);
    }

    public static void addDailyStockData(DailyStockData dailyStockData) {
        insertObject(dailyStockData);
    }

    public static void addDailyStockData(List<DailyStockData> dailyStockData) {
        insertObjects(dailyStockData);
    }

    public static List getDailyStockData() {
        Session session = HibernateUtilities.getSessionFactory().getCurrentSession();
        Criteria criteria = session.createCriteria(DailyStockData.class);
        criteria.add(Restrictions.between("date", "", ""));
        criteria.createAlias("stock", "s", JoinType.INNER_JOIN, Restrictions.eq("s.tickerSymbol", "GOOG"));
        criteria.addOrder(Property.forName("date").asc());
        List results = criteria.list();
        return results;
    }

    public static void initializeDatabase() {
        List<Exchange> exchanges = new ArrayList<>();
        exchanges.add(new Exchange.Builder("NYSE").city("New York").country("USA").currency("USD").build());
        exchanges.add(new Exchange.Builder("NYSE MKT").city("New York").country("USA").currency("USD").build());
        exchanges.add(new Exchange.Builder("NYSE Arca").city("New York").country("USA").currency("USD").build());
        exchanges.add(new Exchange.Builder("NASDAQ").city("New York").country("USA").currency("USD").build());
        addExchanges(exchanges);
        addDataVendor(
                new DataVendor.Builder("Wharton Research Data Services / The Center for Research in Security Prices",
                                       "WRDS/CSRP")
                        .websiteUrl("http://wrds-web.wharton.upenn.edu/wrds/ds/crsp/index.cfm")
                        .build()
        );
    }

    private static void insertObject(Object object) {
        Session session = HibernateUtilities.getSessionFactory().openSession();
        Transaction transaction = session.beginTransaction();
        session.save(object);
        transaction.commit();
        session.close();
    }

    private static void insertObjects(List<?> objects) {
        Session session = HibernateUtilities.getSessionFactory().openSession();
        Transaction transaction = session.beginTransaction();
        int numberOfObjectsSaved = 0;
        for (Object object : objects) {
            session.save(object);
            if (++numberOfObjectsSaved % 20 == 0) { // 20: JDBC batch size used in the current configuration
                session.flush();
                session.clear();
            }
        }
        transaction.commit();
        session.close();
    }

    public static void main(String[] args) {
        initializeDatabase();
        DataImporter.importGICSData();
        HibernateUtilities.getSessionFactory().close();
    }
}