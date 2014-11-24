package org.platanios.trade.data;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.Properties;
import java.util.stream.Stream;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataImporter {
    private static final Logger logger = LogManager.getLogger("Data Importer");
    private static final Properties properties = new Properties();
    private static final String gicsDataFile;
    static {
        try {
            properties.load(ClassLoader.getSystemResourceAsStream("DataImporter.properties"));
            gicsDataFile = properties.getProperty("gicsDataPath");
        } catch (IOException e) {
            logger.error("Could not load the data importer properties.", e);
            throw new RuntimeException(e);
        }
    }

    public static void importGICSData() {
        try (Stream<String> lines = Files.lines(Paths.get(gicsDataFile), Charset.defaultCharset())) {
            Iterator<String> linesIterator = lines.iterator();
            Sector lastSector = null;
            IndustryGroup lastIndustryGroup = null;
            Industry lastIndustry = null;
            while (linesIterator.hasNext()) {
                String[] lineParts = linesIterator.next().split("\t");
                if (!lineParts[0].equals("")) {
                    lastSector = new Sector.Builder(Integer.parseInt(lineParts[0]),
                                                    lineParts[1]).build();
                    DataManager.addSector(lastSector);
                }
                if (!lineParts[2].equals("")) {
                    lastIndustryGroup = new IndustryGroup.Builder(Integer.parseInt(lineParts[2]),
                                                                  lastSector,
                                                                  lineParts[3]).build();
                    DataManager.addIndustryGroup(lastIndustryGroup);
                }
                if (!lineParts[4].equals("")) {
                    lastIndustry = new Industry.Builder(Integer.parseInt(lineParts[4]),
                                                        lastIndustryGroup,
                                                        lineParts[5]).build();
                    DataManager.addIndustry(lastIndustry);
                }
                DataManager.addSubIndustry(new SubIndustry.Builder(Integer.parseInt(lineParts[6]),
                                                                   lastIndustry,
                                                                   lineParts[7]).description(lineParts[8]).build());
            }
        } catch (IOException e) {
            logger.error("Could not load the data importer properties.", e);
            throw new RuntimeException(e);
        }
    }
}
