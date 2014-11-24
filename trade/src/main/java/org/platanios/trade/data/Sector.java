package org.platanios.trade.data;

import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import java.util.Date;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "Sectors",
        catalog = "trade",
        uniqueConstraints = @UniqueConstraint(name = "uk_name", columnNames = "name"))
public class Sector {
    private int gicsId;
    private String name;
    private Date dateTimeCreated;
    private Date dateTimeUpdated;
    private List<IndustryGroup> industryGroups;

    public static class Builder {
        private final int gicsId;
        private final String name;

        public Builder(int gicsId, String name) {
            this.gicsId = gicsId;
            this.name = name;
        }

        public Sector build() {
            Sector sector = new Sector();
            sector.setGicsId(gicsId);
            sector.setName(name);
            return sector;
        }
    }

    protected Sector() {

    }

    @Id
    @Column(name = "gics_id", nullable = false)
    public int getGicsId() {
        return gicsId;
    }

    public void setGicsId(int gicsId) {
        this.gicsId = gicsId;
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

    @OneToMany(mappedBy = "sector", cascade = CascadeType.ALL, orphanRemoval = true)
    public List<IndustryGroup> getIndustryGroups() {
        return industryGroups;
    }

    private void setIndustryGroups(List<IndustryGroup> industryGroups) {
        this.industryGroups = industryGroups;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        Sector that = (Sector) o;

        if (gicsId != that.gicsId)
            return false;
        if (!name.equals(that.name))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = gicsId;
        result = 31 * result + name.hashCode();
        return result;
    }
}
