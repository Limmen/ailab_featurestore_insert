package limmen.github.com.feature_engineering_spark.featuregroup

import org.apache.log4j.{ Level, LogManager, Logger }
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import io.hops.util.Hops
import scala.collection.JavaConversions._
import collection.JavaConverters._

/**
 * Contains logic for computing the hipo_features featuregroup
 */
object HiPoFeatures {

  case class RawHipoRow(
    corporate_id: Int,
    externa_kostnader: Double,
    industry_sector: String,
    netomsettning_1year: Double,
    netomsettning_2year: Double,
    netomsettning_3year: Double)

  case class ParsedHipoRow(
    corporate_id: Int,
    externa_kostnader: Float,
    industry_sector: Long,
    netomsettning_1year: Float,
    netomsettning_2year: Float,
    netomsettning_3year: Float)

  /**
   * Computes the featuregroup
   *
   * @param input path to the input dataset to read (csv)
   * @param output name of the output featuregroup table
   * @param version version of the featuregroup
   * @param partitions number of spark partitions to parallelize the compute on
   * @param logger spark logger
   * @param create boolean flag whether to create new featuregroup or insert into an existing ones
   * @param jobId the id of the hopsworks job
   * @param industrySectorLookup optional pre-computed map with industry sector types categorical to numeric encoding
   */
  def computeFeatures(spark: SparkSession, input: String, featuregroupName: String,
    version: Int, partitions: Int, log: Logger, create: Boolean, jobId: Int,
    industrySectorLookup: Map[String, Long] = null): Unit = {
    log.info(s"Running computeFeatures for featuregroup: ${featuregroupName}")
    val rawDf = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(input).repartition(partitions)
    log.info("Read raw dataframe")
    import spark.implicits._
    val rawDs = rawDf.as[RawHipoRow]
    log.info("Parsed dataframe to dataset")
    val featurestore = Hops.getProjectFeaturestore
    var industrySectorLookupMap: Map[String, Long] = null
    if (industrySectorLookup == null) {
      val industrySectorLookupDf = Hops.getFeaturegroup(spark, "industry_sector_lookup", featurestore, 1)
      val industrySectorLookupList: Array[Row] = industrySectorLookupDf.collect
      val industrySectorLookupMap = industrySectorLookupList.map((row: Row) => {
        row.getAs[String]("industry_sector") -> row.getAs[Long]("id")
      }).toMap
    } else {
      industrySectorLookupMap = industrySectorLookup
    }
    val parsedDs = rawDs.map((hipoRow: RawHipoRow) => {
      val corporateId = hipoRow.corporate_id
      val externaKostnader = hipoRow.externa_kostnader.toFloat
      val industrySector = industrySectorLookupMap(hipoRow.industry_sector)
      val netomsettningOneYear = hipoRow.netomsettning_1year.toFloat
      val netomsettningTwoYear = hipoRow.netomsettning_2year.toFloat
      val netomsettningThreeYear = hipoRow.netomsettning_3year.toFloat
      new ParsedHipoRow(corporateId, externaKostnader, industrySector, netomsettningOneYear, netomsettningTwoYear, netomsettningThreeYear)
    })
    val descriptiveStats = true
    val featureCorr = true
    val featureHistograms = true
    val clusterAnalysis = true
    val statColumns = List[String]().asJava
    val numBins = 20
    val corrMethod = "pearson"
    val numClusters = 5
    val description = "Features on corporate customers"
    val primaryKey = "corporate_id"
    val dependencies = List[String](input).asJava
    if (create) {
      log.info(s"Creating featuregroup $featuregroupName version $version in featurestore $featurestore")
      Hops.createFeaturegroup(spark, parsedDs.toDF, featuregroupName, featurestore, version, description,
        jobId, dependencies, primaryKey, descriptiveStats, featureCorr, featureHistograms, clusterAnalysis,
        statColumns, numBins, corrMethod, numClusters)
      log.info(s"Creation of featuregroup $featuregroupName complete")
    } else {
      log.info(s"Inserting into featuregroup $featuregroupName version $version in featurestore $featurestore")
      Hops.insertIntoFeaturegroup(spark, parsedDs.toDF, featuregroupName, featurestore, version, "overwrite",
        descriptiveStats, featureCorr, featureHistograms, clusterAnalysis, statColumns, numBins, corrMethod,
        numClusters)
      log.info(s"Insertion into featuregroup $featuregroupName complete")
    }
  }
}
