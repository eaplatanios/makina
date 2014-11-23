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
    public void addExchange(Exchange exchange) {
        insertObject(exchange);
    }

    public void addExchanges(List<Exchange> exchanges) {
        insertObjects(exchanges);
    }

    public void addDataVendor(DataVendor dataVendor) {
        insertObject(dataVendor);
    }

    public void addDataVendors(List<DataVendor> dataVendors) {
        insertObjects(dataVendors);
    }

    public void addSector(Sector sector) {
        insertObject(sector);
    }

    public void addSectors(List<Sector> sectors) {
        insertObjects(sectors);
    }

    public void addIndustryGroup(IndustryGroup industryGroup) {
        insertObject(industryGroup);
    }

    public void addIndustryGroups(List<IndustryGroup> industryGroups) {
        insertObjects(industryGroups);
    }

    public void addIndustry(Industry industry) {
        insertObject(industry);
    }

    public void addIndustries(List<Industry> industries) {
        insertObjects(industries);
    }

    public void addSubIndustry(SubIndustry subIndustry) {
        insertObject(subIndustry);
    }

    public void addSubIndustries(List<SubIndustry> subIndustries) {
        insertObjects(subIndustries);
    }

    public void addStock(Stock stock) {
        insertObject(stock);
    }

    public void addStocks(List<Stock> stocks) {
        insertObjects(stocks);
    }

    public void addDailyStockData(DailyStockData dailyStockData) {
        insertObject(dailyStockData);
    }

    public void addDailyStockData(List<DailyStockData> dailyStockData) {
        insertObjects(dailyStockData);
    }

    public List getDailyStockData() {
        Session session = HibernateUtilities.getSessionFactory().getCurrentSession();
        Criteria criteria = session.createCriteria(DailyStockData.class);
        criteria.add(Restrictions.between("date", "", ""));
        criteria.createAlias("stock", "s", JoinType.INNER_JOIN, Restrictions.eq("s.tickerSymbol", "GOOG"));
        criteria.addOrder(Property.forName("date").asc());
        List results = criteria.list();
        return results;
    }

    public void initializeDatabase() {
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

    private void insertObject(Object object) {
        Session session = HibernateUtilities.getSessionFactory().openSession();
        Transaction transaction = session.beginTransaction();
        session.save(object);
        transaction.commit();
        session.close();
    }

    private void insertObjects(List<?> objects) {
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
        DataManager dataManager = new DataManager();
        dataManager.initializeDatabase();
        HibernateUtilities.getSessionFactory().close();
    }
}