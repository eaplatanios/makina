package org.platanios.trade.data;

import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import java.util.Date;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "SubIndustries",
        catalog = "trade",
        uniqueConstraints = @UniqueConstraint(name = "uk_name", columnNames = "name"))
public class SubIndustry {
    private int gicsId;
    private Industry industry;
    private String name;
    private String description;
    private Date dateTimeCreated;
    private Date dateTimeUpdated;

    public static class Builder {
        private final int gicsId;
        private final Industry industry;
        private final String name;

        private String description;

        public Builder(int gicsId, Industry industry, String name) {
            this.gicsId = gicsId;
            this.industry = industry;
            this.name = name;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public SubIndustry build() {
            SubIndustry subIndustry = new SubIndustry();
            subIndustry.setGicsId(gicsId);
            subIndustry.setIndustry(industry);
            subIndustry.setName(name);
            subIndustry.setDescription(description);
            return subIndustry;
        }
    }

    protected SubIndustry() {

    }

    @Id
    @Column(name = "gics_id", nullable = false)
    public int getGicsId() {
        return gicsId;
    }

    public void setGicsId(int gicsId) {
        this.gicsId = gicsId;
    }

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "industry_gics_id", foreignKey = @ForeignKey(name = "fk_industry"))
    @NotNull
    public Industry getIndustry() {
        return industry;
    }

    public void setIndustry(Industry industry) {
        this.industry = industry;
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

    @Basic
    @Column(name = "description")
    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
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

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        SubIndustry that = (SubIndustry) o;

        if (gicsId != that.gicsId)
            return false;
        if (!industry.equals(that.industry))
            return false;
        if (!name.equals(that.name))
            return false;
        if (description != null ? !description.equals(that.description) : that.description != null)
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = gicsId;
        result = 31 * result + industry.hashCode();
        result = 31 * result + name.hashCode();
        result = 31 * result + (description != null ? description.hashCode() : 0);
        return result;
    }
}
