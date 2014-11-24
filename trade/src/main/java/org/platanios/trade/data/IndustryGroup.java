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
@Table(name = "IndustryGroups",
        catalog = "trade",
        uniqueConstraints = @UniqueConstraint(name = "uk_name", columnNames = "name"))
public class IndustryGroup {
    private int gicsId;
    private Sector sector;
    private String name;
    private Date dateTimeCreated;
    private Date dateTimeUpdated;
    private List<Industry> industries;

    public static class Builder {
        private final int gicsId;
        private final Sector sector;
        private final String name;

        public Builder(int gicsId, Sector sector, String name) {
            this.gicsId = gicsId;
            this.sector = sector;
            this.name = name;
        }

        public IndustryGroup build() {
            IndustryGroup industryGroup = new IndustryGroup();
            industryGroup.setGicsId(gicsId);
            industryGroup.setSector(sector);
            industryGroup.setName(name);
            return industryGroup;
        }
    }

    protected IndustryGroup() {

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
    @JoinColumn(name = "sector_gics_id", nullable = false, foreignKey = @ForeignKey(name = "fk_sector"))
    @NotNull
    public Sector getSector() {
        return sector;
    }

    public void setSector(Sector sector) {
        this.sector = sector;
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

    @OneToMany(mappedBy = "industryGroup", cascade = CascadeType.ALL, orphanRemoval = true)
    public List<Industry> getIndustries() {
        return industries;
    }

    private void setIndustries(List<Industry> industries) {
        this.industries = industries;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        IndustryGroup that = (IndustryGroup) o;

        if (gicsId != that.gicsId)
            return false;
        if (!sector.equals(that.sector))
            return false;
        if (!name.equals(that.name))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = gicsId;
        result = 31 * result + sector.hashCode();
        result = 31 * result + name.hashCode();
        return result;
    }
}
