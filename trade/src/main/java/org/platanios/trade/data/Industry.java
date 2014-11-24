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
@Table(name = "Industries",
        catalog = "trade",
        uniqueConstraints = @UniqueConstraint(name = "uk_name", columnNames = "name"))
public class Industry {
    private int gicsId;
    private IndustryGroup industryGroup;
    private String name;
    private Date dateTimeCreated;
    private Date dateTimeUpdated;
    private List<SubIndustry> subIndustries;

    public static class Builder {
        private final int gicsId;
        private final IndustryGroup industryGroup;
        private final String name;

        public Builder(int gicsId, IndustryGroup industryGroup, String name) {
            this.gicsId = gicsId;
            this.industryGroup = industryGroup;
            this.name = name;
        }

        public Industry build() {
            Industry industry = new Industry();
            industry.setGicsId(gicsId);
            industry.setIndustryGroup(industryGroup);
            industry.setName(name);
            return industry;
        }
    }

    protected Industry() {

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
    @JoinColumn(name = "industry_group_gics_id", nullable = false, foreignKey = @ForeignKey(name = "fk_industry_group"))
    @NotNull
    public IndustryGroup getIndustryGroup() {
        return industryGroup;
    }

    public void setIndustryGroup(IndustryGroup industryGroup) {
        this.industryGroup = industryGroup;
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

    @OneToMany(mappedBy = "industry", cascade = CascadeType.ALL, orphanRemoval = true)
    public List<SubIndustry> getSubIndustries() {
        return subIndustries;
    }

    private void setSubIndustries(List<SubIndustry> subIndustries) {
        this.subIndustries = subIndustries;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        Industry that = (Industry) o;

        if (gicsId != that.gicsId)
            return false;
        if (!industryGroup.equals(that.industryGroup))
            return false;
        if (!name.equals(that.name))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = gicsId;
        result = 31 * result + industryGroup.hashCode();
        result = 31 * result + name.hashCode();
        return result;
    }
}