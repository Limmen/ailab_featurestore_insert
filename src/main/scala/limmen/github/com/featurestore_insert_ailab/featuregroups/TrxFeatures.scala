package limmen.github.com.feature_engineering_spark.featuregroup

import org.apache.log4j.{ Level, LogManager, Logger }
import java.sql.Timestamp
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import io.hops.util.Hops
import scala.collection.JavaConversions._
import collection.JavaConverters._

/**
 * Contains logic for computing the trx_features featuregroup
 */
object TrxFeatures {

  case class RawTrx(
    cust_id_in: Int,
    cust_id_out: Int,
    trx_amount: Double,
    trx_bankid: Int,
    trx_clearinnum: Int,
    trx_country: String,
    trx_date: String,
    trx_type: String,
    trx_id: Int)

  case class TrxFeature(
    cust_id_in: Int,
    trx_type: Long,
    trx_date: Timestamp,
    trx_amount: Float,
    trx_bankid: Int,
    cust_id_out: Int,
    trx_clearingnum: Int,
    trx_country: Long,
    trx_id: Int)

  val formatter = new java.text.SimpleDateFormat("yyyy-MM-dd")

  /**
   * Computes the featuregroup
   *
   * @param input path to the input dataset to read (csv)
   * @param output name of the output featuregroup table
   * @param version version of the featuregroup
   * @param partitions number of spark partitions to parallelize the compute on
   * @param log spark logger
   * @param create boolean flag whether to create new featuregroup or insert into an existing ones
   * @param jobId the id of the hopsworks job
   * @param trxTypeLookup optional pre-computed map with transaction types categorical to numeric encoding
   * @param countryTypeLookup optional pre-computed map with country types categorical to numeric encoding
   */
  def computeFeatures(spark: SparkSession, input: String, featuregroupName: String,
    version: Int, partitions: Int, log: Logger, create: Boolean, jobId: Int,
    countryLookup: Map[String, Long] = null, trxTypeLookup: Map[String, Long] = null): Unit = {
    log.info(s"Running computeFeatures for featuregroup: ${featuregroupName}")
    val rawDf = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(input).repartition(partitions)
    log.info("Read raw dataframe")
    import spark.implicits._
    val rawDs = rawDf.as[RawTrx]
    log.info("Parsed dataframe to dataset")
    val featurestore = Hops.getProjectFeaturestore
    var countryLookupMap: Map[String, Long] = null
    if (countryLookup == null) {
      val countryLookupDf = Hops.getFeaturegroup(spark, "country_lookup", featurestore, 1)
      val countryLookupList: Array[Row] = countryLookupDf.collect
      countryLookupMap = countryLookupList.map((row: Row) => {
        row.getAs[String]("trx_country") -> row.getAs[Long]("id")
      }).toMap
    } else {
      countryLookupMap = countryLookup
    }
    var trxTypeLookupMap: Map[String, Long] = null
    if (trxTypeLookup == null) {
      val trxTypeLookupDf = Hops.getFeaturegroup(spark, "trx_type_lookup", featurestore, 1)
      val trxTypeLookupList: Array[Row] = trxTypeLookupDf.collect
      val trxTypeLookupMap = trxTypeLookupList.map((row: Row) => {
        row.getAs[String]("trx_type") -> row.getAs[Long]("id")
      }).toMap
    } else {
      trxTypeLookupMap = trxTypeLookup
    }
    val parsedDs = rawDs.map((trx: RawTrx) => {
      val cust_id_in = trx.cust_id_in
      val cust_id_out = trx.cust_id_out
      val trxDate = new Timestamp(formatter.parse(trx.trx_date).getTime)
      val trx_amount = trx.trx_amount.toFloat
      val trx_bankid = trx.trx_bankid
      val trx_clearingnum = trx.trx_clearinnum
      val trx_country = countryLookupMap(trx.trx_country)
      val trx_type = trxTypeLookupMap(trx.trx_type)
      val trx_id = trx.trx_id
      TrxFeature(cust_id_in, trx_type, trxDate, trx_amount, trx_bankid, cust_id_out, trx_clearingnum, trx_country, trx_id)
    })
    val descriptiveStats = true
    val featureCorr = true
    val featureHistograms = true
    val clusterAnalysis = true
    val statColumns = List[String]().asJava
    val numBins = 20
    val corrMethod = "pearson"
    val numClusters = 5
    val description = "Features for single transactions"
    val primaryKey = "trx_id"
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
