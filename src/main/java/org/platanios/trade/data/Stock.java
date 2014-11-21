package org.platanios.trade.data;

import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.NaturalId;
import org.hibernate.annotations.UpdateTimestamp;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import java.util.Date;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "Stocks", catalog = "trade")
public class Stock {
    private long id;
    private String cusipId;
    private String cikId;
    private String tickerSymbol;
    private Exchange primaryExchange;
    private String name;
    private SubIndustry subIndustry;
    private Date dateTimeCreated;
    private Date dateTimeUpdated;
    private List<DailyStockData> dailyData;

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    public long getId() {
        return id;
    }

    private void setId(long id) {
        this.id = id;
    }

    @NaturalId
    @Basic
    @Column(name = "cusip_id", unique = true)
    public String getCusipId() {
        return cusipId;
    }

    public void setCusipId(String cusipId) {
        this.cusipId = cusipId;
    }

    @NaturalId
    @Basic
    @Column(name = "cik_id", unique = true)
    public String getCikId() {
        return cikId;
    }

    public void setCikId(String cikId) {
        this.cikId = cikId;
    }

    @NaturalId
    @Basic
    @Column(name = "ticker_symbol", unique = true, nullable = false)
    @NotNull
    public String getTickerSymbol() {
        return tickerSymbol;
    }

    public void setTickerSymbol(String tickerSymbol) {
        this.tickerSymbol = tickerSymbol;
    }

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "primary_exchange_id", nullable = false)
    @NotNull
    public Exchange getPrimaryExchange() {
        return primaryExchange;
    }

    public void setPrimaryExchange(Exchange primaryExchange) {
        this.primaryExchange = primaryExchange;
    }

    @Basic
    @Column(name = "name", nullable = false)
    @NotNull
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "sub_industry_id")
    public SubIndustry getSubIndustry() {
        return subIndustry;
    }

    public void setSubIndustry(SubIndustry subIndustry) {
        this.subIndustry = subIndustry;
    }

    @Temporal(TemporalType.TIMESTAMP)
    @Column(name = "date_time_created")
    @CreationTimestamp
    public Date getDateTimeCreated() {
        return dateTimeCreated;
    }

    private void setDateTimeCreated(Date dateTimeCreated) {
        this.dateTimeCreated = dateTimeCreated;
    }

    @Temporal(TemporalType.TIMESTAMP)
    @Column(name = "date_time_updated")
    @UpdateTimestamp
    public Date getDateTimeUpdated() {
        return dateTimeUpdated;
    }

    private void setDateTimeUpdated(Date dateTimeUpdated) {
        this.dateTimeUpdated = dateTimeUpdated;
    }

    @OneToMany(mappedBy = "stock", cascade = CascadeType.ALL, orphanRemoval = true)
    public List<DailyStockData> getDailyData() {
        return dailyData;
    }

    private void setDailyData(List<DailyStockData> dailyData) {
        this.dailyData = dailyData;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        Stock that = (Stock) o;

        if (id != that.id)
            return false;
        if (cusipId != null ? !cusipId.equals(that.cusipId) : that.cusipId != null)
            return false;
        if (cikId != null ? !cikId.equals(that.cikId) : that.cikId != null)
            return false;
        if (!tickerSymbol.equals(that.tickerSymbol))
            return false;
        if (!primaryExchange.equals(that.primaryExchange))
            return false;
        if (!name.equals(that.name))
            return false;
        if (subIndustry != null ? !subIndustry.equals(that.subIndustry) : that.subIndustry != null)
            return false;
        if (!dateTimeCreated.equals(that.dateTimeCreated))
            return false;
        if (!dateTimeUpdated.equals(that.dateTimeUpdated))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = (int) (id ^ (id >>> 32));
        result = 31 * result + (cusipId != null ? cusipId.hashCode() : 0);
        result = 31 * result + (cikId != null ? cikId.hashCode() : 0);
        result = 31 * result + tickerSymbol.hashCode();
        result = 31 * result + primaryExchange.hashCode();
        result = 31 * result + name.hashCode();
        result = 31 * result + (subIndustry != null ? subIndustry.hashCode() : 0);
        result = 31 * result + dateTimeCreated.hashCode();
        result = 31 * result + dateTimeUpdated.hashCode();
        return result;
    }
}
