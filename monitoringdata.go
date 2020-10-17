package monitoringdata

import (
	"strconv"
	"strings"

	"sort"

	"gonum.org/v1/gonum/stat"

	//"encoding/json"
	"math"
)

type dataStatistics struct {
	Name                     string
	Datatype                 string
	Mean                     float64
	Median                   float64
	Q1                       float64
	Q3                       float64
	IQR                      float64
	STDEV                    float64
	Max                      float64
	Min                      float64
	UniqueItemsCount         int
	UniqueItems              interface{}
	UniqueItemsCountValues   []float64
	VariableType             string
	DistType                 string
	MissingDataCount         int
	QuantileDataDistribution []float64
	UniqueItemCountAndValues map[string]int
}

var colDatacollectionStructOriginal map[string]dataStatistics
var colDatacollectionStructCurrent map[string]dataStatistics

// GetStatisticsOnData function does the following
// It takes as input the original training data and the current data
// Build elementary statistics on the individual datasets
// Calculates the Population Stability Index
func GetStatisticsOnData(originalRecord [][]string, currentRecord [][]string) map[string]interface{} {

	// Open the file
	/*csvfile, err := os.Open("titanicTrain.csv")
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}

	// Parse the file
	r := csv.NewReader(csvfile)

	record, err := r.ReadAll()*/

	columnNumbers := len(originalRecord[0])
	colNames := make([]string, columnNumbers)
	totalNumberofRecordsInOriginal := float64(len(originalRecord) - 1)
	totalNumberofRecordsInCurrent := float64(len(originalRecord) - 1)
	dataSummaryAndStatistics := make(map[string]interface{}, 0)

	for i := 0; i < columnNumbers; i++ {
		colNames[i] = originalRecord[0][i]
	}

	colDatacollectionStructOriginal = processAndCollectStatsOnData(colNames, originalRecord, true)
	colDatacollectionStructCurrent = processAndCollectStatsOnData(colNames, currentRecord, false)

	populationStabilityIndexValues := calculateOverallPopulationStabilityIndex(colNames, totalNumberofRecordsInOriginal, totalNumberofRecordsInCurrent)

	dataSummaryAndStatistics["OriginalData"] = colDatacollectionStructOriginal
	dataSummaryAndStatistics["CurrentData"] = colDatacollectionStructCurrent
	dataSummaryAndStatistics["PopulationStabilityIndexValues"] = populationStabilityIndexValues

	//retJSONString, err := json.Marshal(dataSummaryAndStatistics)

	//fmt.Println(string(retJSONString))

	return dataSummaryAndStatistics

}

func calculateOverallPopulationStabilityIndex(colNames []string, totalNumberofRecordsInOriginal float64, totalNumberofRecordsInCurrent float64) map[string]float64 {

	populationStabilityIndexValues := make(map[string]float64, len(colNames))

	for _, colName := range colNames {
		if colDatacollectionStructOriginal[colName].DistType == "Non Unique" {
			if colDatacollectionStructOriginal[colName].VariableType == "Continuous" {
				psi := calculatePopulationStabilityIndexContinuous(colDatacollectionStructOriginal[colName].QuantileDataDistribution, colDatacollectionStructCurrent[colName].QuantileDataDistribution, totalNumberofRecordsInOriginal, totalNumberofRecordsInCurrent)
				populationStabilityIndexValues[colName] = psi
			} else if colDatacollectionStructOriginal[colName].VariableType == "Categorical" || colDatacollectionStructOriginal[colName].VariableType == "Discrete" {
				psi := calculatePopulationStabilityIndexCategorical(colDatacollectionStructOriginal[colName].UniqueItemCountAndValues, colDatacollectionStructCurrent[colName].UniqueItemCountAndValues, totalNumberofRecordsInOriginal, totalNumberofRecordsInCurrent)
				populationStabilityIndexValues[colName] = psi
			}
		}
	}

	return populationStabilityIndexValues

}

func calculatePopulationStabilityIndexCategorical(originalData map[string]int, currentData map[string]int, totalOriginal float64, totalCurrent float64) float64 {

	var psi float64

	for strKey, data := range currentData {
		actual := float64(originalData[strKey]) / float64(totalOriginal)
		expected := float64(data) / float64(totalCurrent)
		psi = psi + ((actual - expected) * math.Log(actual/expected))
	}

	return psi
}

func calculatePopulationStabilityIndexContinuous(originalData []float64, currentData []float64, totalOriginal float64, totalCurrent float64) float64 {

	var psi float64

	for i := 0; i < len(originalData); i++ {
		actual := originalData[i] / totalOriginal
		expected := currentData[i] / totalCurrent
		psi = psi + ((actual - expected) * math.Log(actual/expected))
	}

	return psi
}

func processAndCollectStatsOnData(colNames []string, record [][]string, isOriginal bool) map[string]dataStatistics {

	colDatacollectionStruct := make(map[string]dataStatistics)

	totalNumberofRecords := len(record) - 1
	columnNumbers := len(record[0])
	colDataTypes := make(map[string]string)

	for i := 0; i < columnNumbers; i++ {
		colNames[i] = record[0][i]
	}

	for i := 0; i < columnNumbers; i++ {
		if isNumeric(record[1][i]) {
			if isInt(record[1][i]) {
				colDataTypes[colNames[i]] = "int64"
			} else {
				colDataTypes[colNames[i]] = "float64"
			}
		} else {
			colDataTypes[colNames[i]] = "string"
		}
	}

	for col := 0; col < columnNumbers; col++ {

		var tempColDataValues []float64
		var tempColDataValuesString []string
		var missingDataCount int

		if colDataTypes[colNames[col]] != "string" {
			for row := 1; row <= totalNumberofRecords; row++ {

				if strings.TrimSpace(record[row][col]) == "" {
					missingDataCount = missingDataCount + 1
				}
				tempColDataValues = append(tempColDataValues, convertToFloat64(record[row][col]))
			}
			//colDatacollectionFloat64[colNames[col]] = tempColDataValues

			retDS := getStatsOnDataFloat64(colNames[col], tempColDataValues, colDataTypes[colNames[col]], totalNumberofRecords, missingDataCount, isOriginal)

			colDatacollectionStruct[colNames[col]] = retDS

		} else {
			for row := 1; row <= totalNumberofRecords; row++ {

				if strings.TrimSpace(record[row][col]) == "" {
					record[row][col] = strings.TrimSpace(record[row][col])
				}

				tempColDataValuesString = append(tempColDataValuesString, record[row][col])
			}
			//colDatacollectionFloat64[colNames[col]] = tempColDataValues

			retDS := getStatsOnDataString(colNames[col], tempColDataValuesString, colDataTypes[colNames[col]], totalNumberofRecords)

			colDatacollectionStruct[colNames[col]] = retDS

		}
	}

	return colDatacollectionStruct

}

func countValuesInRange(data []float64, Q1 float64, Q2 float64, Q3 float64, ignoreZeroValues bool) []float64 {

	countedValuesInQuantile := make([]float64, 0)

	dataCountQ1 := 0.0
	dataCountQ2 := 0.0
	dataCountQ3 := 0.0
	dataCountQ4 := 0.0

	for _, num := range data {

		if ignoreZeroValues && num == 0 {
			continue
		}

		if num <= Q1 {
			dataCountQ1 = dataCountQ1 + 1
		} else if num > Q1 && num <= Q2 {
			dataCountQ2 = dataCountQ2 + 1
		} else if num > Q2 && num <= Q3 {
			dataCountQ3 = dataCountQ3 + 1
		} else if num > Q3 {
			dataCountQ4 = dataCountQ4 + 1
		}
	}

	countedValuesInQuantile = append(countedValuesInQuantile, dataCountQ1, dataCountQ2, dataCountQ3, dataCountQ4)

	return countedValuesInQuantile

}

func countUniqueItems(data []float64) ([]float64, []interface{}, int) {

	dataCount := make(map[float64]int)
	uniqueItems := make([]interface{}, 0)
	uniqueItemsCountValues := make([]float64, 0)

	for _, num := range data {
		dataCount[num] = dataCount[num] + 1
	}

	for key, countValue := range dataCount {
		uniqueItems = append(uniqueItems, key)
		uniqueItemsCountValues = append(uniqueItemsCountValues, float64(countValue))
	}

	return uniqueItemsCountValues, uniqueItems, len(dataCount)
}

func countUniqueItemsString(data []string) ([]float64, []interface{}, int, int, map[string]int) {

	dataCount := make(map[string]int)
	uniqueItems := make([]interface{}, 0)
	uniqueItemsCountValues := make([]float64, 0)

	for _, num := range data {
		dataCount[num] = dataCount[num] + 1
	}

	for key, countValue := range dataCount {
		uniqueItems = append(uniqueItems, key)
		uniqueItemsCountValues = append(uniqueItemsCountValues, float64(countValue))
	}

	missingValueCount := dataCount[""]

	return uniqueItemsCountValues, uniqueItems, len(dataCount), missingValueCount, dataCount
}

func isDiscrete(s string) bool {
	_, err := strconv.ParseFloat(s, 64)
	return err == nil
}

func isNumeric(s string) bool {
	_, err := strconv.ParseFloat(s, 64)
	return err == nil
}

func isInt(s string) bool {
	_, err := strconv.ParseInt(s, 10, 64)
	return err == nil
}

func convertToInt64(s string) int64 {
	value, _ := strconv.ParseInt(s, 10, 64)
	return value
}

func convertToFloat64(s string) float64 {
	value, _ := strconv.ParseFloat(s, 64)
	return value
}

func getStatsOnDataString(name string, data []string, datatype string, totalNumberofRecords int) dataStatistics {

	var ds dataStatistics
	var distType string

	uniqueItemsCountValues, uniqueItems, countedUniqueItems, missingValueCount, dataCount := countUniqueItemsString(data)

	if (float64(countedUniqueItems) / float64(totalNumberofRecords)) > 0.9 {
		uniqueItemsCountValues = nil
		uniqueItems = nil
		distType = "Unique"
		dataCount = nil
	} else {
		distType = "Non Unique"

	}
	ds = dataStatistics{
		Name:                     name,
		Datatype:                 datatype,
		UniqueItemsCount:         countedUniqueItems,
		UniqueItems:              uniqueItems,
		VariableType:             "Categorical",
		UniqueItemsCountValues:   uniqueItemsCountValues,
		DistType:                 distType,
		MissingDataCount:         missingValueCount,
		UniqueItemCountAndValues: dataCount,
	}
	return ds
}

func getStatsOnDataFloat64(name string, data []float64, datatype string, totalNumberofRecords int, missingDataCount int, isOriginal bool) dataStatistics {

	// Making an assumption if the number of unique items is less than 5% of the overall data then
	// its Categorical

	var ds dataStatistics
	var Q1, Q2, Q3 float64

	sort.Float64s(data)

	uniqueItemsCountValues, uniqueItems, countedUniqueItems := countUniqueItems(data)

	//if (float64(countedUniqueItems) / float64(totalNumberofRecords)) > 0.98 {
	if countedUniqueItems == totalNumberofRecords {
		ds = dataStatistics{
			Name:             name,
			Datatype:         datatype,
			UniqueItemsCount: countedUniqueItems,
			VariableType:     "Continuous",
			DistType:         "Unique",
		}
	} else if float64(countedUniqueItems) > 0.02*float64(totalNumberofRecords) {

		if isOriginal {
			Q1 = stat.Quantile(0.25, 4, data, nil)
			Q2 = stat.Quantile(0.50, 4, data, nil)
			Q3 = stat.Quantile(0.75, 4, data, nil)
		} else {
			Q1 = colDatacollectionStructOriginal[name].Q1
			Q2 = colDatacollectionStructOriginal[name].Median
			Q3 = colDatacollectionStructOriginal[name].Q3
		}
		ignoreZeroValues := true

		quantileDataDistribution := countValuesInRange(data, Q1, Q2, Q3, ignoreZeroValues)

		ds = dataStatistics{
			Name:                     name,
			Datatype:                 datatype,
			Mean:                     stat.Mean(data, nil),
			Q1:                       Q1,
			Q3:                       Q3,
			IQR:                      Q3 - Q1,
			Median:                   Q2,
			STDEV:                    stat.StdDev(data, nil),
			Max:                      stat.Quantile(1, 4, data, nil),
			Min:                      stat.Quantile(0, 4, data, nil),
			UniqueItemsCount:         countedUniqueItems,
			UniqueItems:              uniqueItems,
			VariableType:             "Continuous",
			UniqueItemsCountValues:   uniqueItemsCountValues,
			DistType:                 "Non Unique",
			MissingDataCount:         missingDataCount,
			QuantileDataDistribution: quantileDataDistribution,
		}
	} else {
		ds = dataStatistics{
			Name:                   name,
			Datatype:               datatype,
			UniqueItemsCount:       countedUniqueItems,
			UniqueItems:            uniqueItems,
			VariableType:           "Discrete",
			UniqueItemsCountValues: uniqueItemsCountValues,
			DistType:               "Non Unique",
			MissingDataCount:       missingDataCount,
		}
	}

	return ds
}
