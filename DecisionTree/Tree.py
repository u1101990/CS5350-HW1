import math

#Parameters
CSVfile = 'train.csv'
labels = ["unacc", "acc", "good", "vgood"]
attributes = ["buying", "maint", "doors", "persons", "lug_boot", "safety"] #in the order that elements appear
attribParams = [["vhigh", "high", "med", "low"], ["vhigh", "high", "med", "low"], ["2", "3", "4", "5more"], ["2", "4", "more"], ["small", "med", "big"], ["low", "med", "high"]]
columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]
 
testSamples = []
trainSamples = []

#Load Training Set
with open(CSVfile, 'r') as f:
    for line in f:
        terms = line.strip().split(',')
        trainSamples.append(terms)

count = len(trainSamples) #Total number of training examples
#The statistics of a column. Each stat is an array containing all the posibilities
#Results are the associations with a lable [unacc, acc, good, vgood]
#           vhigh, high, med, low
buyingStats = [0, 0, 0, 0]
buyingResults = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
#           vhigh, high, med, low
maintStats = [0, 0, 0, 0]
maintResults = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
#           2, 3, 4, 5more
doorsStats = [0, 0, 0, 0]
doorsResults = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
#           2, 4, more
personsStats = [0, 0, 0]
personsResults = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
#           small, med, big
lug_bootStats = [0, 0, 0]
lug_bootResults = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
#           low, med, high
safetyStats = [0, 0, 0]
safetyResults = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
#           unacc, acc, good, vgood
labelsStats = [0, 0, 0, 0]


#Sumarize Statistics
for sample in trainSamples:

    label = 0

    if sample[6] == 'unacc':
        labelsStats[0] += 1
        label = 0
    elif sample[6] == 'acc':
        labelsStats[1] += 1
        label = 1
    elif sample[6] == 'good':
        labelsStats[2] += 1
        label = 2
    elif sample[6] == 'vgood':
        labelsStats[3] += 1
        label = 3

    if sample[0] == 'vhigh':
        buyingStats[0] += 1
        buyingResults[0][label] += 1
    elif sample[0] == 'high':
        buyingStats[1] += 1
        buyingResults[1][label] += 1
    elif sample[0] == 'med':
        buyingStats[2] += 1
        buyingResults[2][label] += 1
    elif sample[0] == 'low':
        buyingStats[3] += 1
        buyingResults[3][label] += 1

    if sample[1] == 'vhigh':
        maintStats[0] += 1
        maintResults[0][label] += 1
    elif sample[1] == 'high':
        maintStats[1] += 1
        maintResults[1][label] += 1
    elif sample[1] == 'med':
        maintStats[2] += 1
        maintResults[2][label] += 1
    elif sample[1] == 'low':
        maintStats[3] += 1
        maintResults[3][label] += 1

    if sample[2] == '2':
        doorsStats[0] += 1
        doorsResults[0][label] += 1
    elif sample[2] == '3':
        doorsStats[1] += 1
        doorsResults[1][label] += 1
    elif sample[2] == '4':
        doorsStats[2] += 1
        doorsResults[2][label] += 1
    elif sample[2] == '5more':
        doorsStats[3] += 1
        doorsResults[3][label] += 1

    if sample[3] == '2':
        personsStats[0] += 1
        personsResults[0][label] += 1
    elif sample[3] == '4':
        personsStats[1] += 1
        personsResults[1][label] += 1
    elif sample[3] == 'more':
        personsStats[2] += 1
        personsResults[2][label] += 1

    if sample[4] == 'small':
        lug_bootStats[0] += 1
        lug_bootResults[0][label] += 1
    elif sample[4] == 'med':
        lug_bootStats[1] += 1
        lug_bootResults[1][label] += 1
    elif sample[4] == 'big':
        lug_bootStats[2] += 1
        lug_bootResults[2][label] += 1

    if sample[5] == 'low':
        safetyStats[0] += 1
        safetyResults[0][label] += 1
    elif sample[5] == 'med':
        safetyStats[1] += 1
        safetyResults[1][label] += 1
    elif sample[5] == 'high':
        safetyStats[2] += 1
        safetyResults[2][label] += 1



#the "log Squre"
def logSquare(prob):
    if prob != 0:
        return prob * math.log(prob, 2)
    
    return 0

#returns the overall entropy of the entire training set.
def baseEntropy():
    choices = len(labelsStats)
    probabilities = []
    for i in range(choices):
        probabilities += [labelsStats[i] / count]

    total = 0

    for i in range(choices):
        total += logSquare(probabilities[i])

    total *= -1

    return total
        

#Returns the entropy of one set.
#Takes in one option of a subset (such as just the vhigh results of buyers)
#Size is the sum of the results array.
def calculateEntropy(results, size):
    choices = len(results)
    probabilities = []
    for i in range(choices):
        probabilities += [(results[i]) / size]

    total = 0

    for i in range(choices):
        total += logSquare(probabilities[i])

    total *= -1

    return total

def calculateInformationGain(votes, results):
    systemEntropy = baseEntropy()

    #Get the entropy of all subsets
    choices = len(votes)
    subsetEntropy = []
    for i in range(choices):
        subsetEntropy += [calculateEntropy(results[i], votes[i])]
    
    #Get the ratios of the sets
    subsetPercentiles = []
    for i in range(choices):
        subsetPercentiles += [votes[i] / count]

    totalEntropy = 0
    for i in range(choices):
        totalEntropy += subsetEntropy[i] * subsetPercentiles[i]

    informationGain = systemEntropy - totalEntropy

    return informationGain

#calculates the majority error of a set.
def calculateMajorityError(votes, results):
    #For Each Subset
    choices = len(votes)
    ME = 0
    for i in range(len(votes)):
        #Get the most common result of said subset
        mostCommonIndex = results[i].index(max(results[i]))
        #Get the percentile of the most common result
        subsetPercentiles = []
        for i in range(choices):
            subsetPercentiles += [votes[i] / count]

        purity = max(subsetPercentiles)
        #Get subsetSize * (1 - mostCommonPercentile) 
        accuracy = 1 - purity
        value = votes[i] * accuracy
        ME += value

    #Sum these, then divide by total results

    ME = ME / count

    return ME

def calculateGiniIndex(votes, results):
    print("Gini Index not yet implemented.")

print('Buying:', buyingStats)
print('Maint:', maintStats)
print('Doors:', doorsStats)
print('Persons:', personsStats)
print('LugBoots:', lug_bootStats)
print('Safety:', safetyStats)
print('Labels:', labelsStats)
print('Buying:', buyingResults)
print('Maint:', maintResults)
print('Doors:', doorsResults)
print('Persons:', personsResults)
print('LugBoots:', lug_bootResults)
print('Safety:', safetyResults)

buyGain = calculateInformationGain(buyingStats, buyingResults)
maintGain = calculateInformationGain(maintStats, maintResults)
DoorsGain = calculateInformationGain(doorsStats, doorsResults)
PersonsGain = calculateInformationGain(personsStats, personsResults)
LugBootsGain = calculateInformationGain(lug_bootStats, lug_bootResults)
SafetyGain = calculateInformationGain(safetyStats, safetyResults)

print('Starting Entropy', baseEntropy())
print('Buying Gain:', buyGain)
print('Maint Gain:', maintGain)
print('Doors Gain:', DoorsGain)
print('Persons Gain:', PersonsGain)
print('LugBoots Gain:', LugBootsGain)
print('Safety:Gain', SafetyGain)

buyME = calculateMajorityError(buyingStats, buyingResults)
maintME = calculateMajorityError(maintStats, maintResults)
DoorsME = calculateMajorityError(doorsStats, doorsResults)
PersonsME = calculateMajorityError(personsStats, personsResults)
LugBootsME = calculateMajorityError(lug_bootStats, lug_bootResults)
SafetyME = calculateMajorityError(safetyStats, safetyResults)

print('Buying ME:', buyME)
print('Maint ME:', maintME)
print('Doors ME:', DoorsME)
print('Persons ME:', PersonsME)
print('LugBoots ME:', LugBootsME)
print('Safety:ME', SafetyME)
