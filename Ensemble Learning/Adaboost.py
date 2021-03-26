import math
import statistics

#Parameters
CSVfile = 'train.csv'
#labels = ["unacc", "acc", "good", "vgood"]
#attributes = ["buying", "maint", "doors", "persons", "lug_boot", "safety"] #in the order that elements appear
#attribParams = [["vhigh", "high", "med", "low"], ["vhigh", "high", "med", "low"], ["2", "3", "4", "5more"], ["2", "4", "more"], ["small", "med", "big"], ["low", "med", "high"]]
#columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]

labels = ["yes", "no"]
attributes = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"] #in the order that elements appear
attribParams = [
["yes", "no"], 
["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"], 
["married","divorced","single"], 
["unknown","secondary","primary","tertiary"], 
["yes", "no"], 
["yes", "no"], 
["yes", "no"],
["yes", "no"],
["unknown","telephone","cellular"],
["yes", "no"],
["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
["yes", "no"],
["yes", "no"],
["yes", "no"],
["yes", "no"],
["unknown","other","failure","success"],
]
columns = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "label"]
 
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
#           yes, no (high or low)
ageStats = [0, 0]
ageResults = [[0, 0], [0, 0]]
#           "admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"
jobStats = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
jobResults = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
#           "married","divorced","single"
maritalStats = [0, 0, 0]
maritalResults = [[0, 0], [0, 0], [0, 0]]
#           "unknown","secondary","primary","tertiary"
educationStats = [0, 0, 0, 0]
educationResults = [[0, 0], [0, 0], [0, 0], [0, 0]]
#           yes, no (high or low)
defaultStats = [0, 0]
defaultResults = [[0, 0], [0, 0]]
#           yes, no (high or low)
balanceStats = [0, 0]
balanceResults = [[0, 0], [0, 0]]
#           yes, no (high or low)
housingStats = [0, 0]
housingResults = [[0, 0], [0, 0]]
#           yes, no (high or low)
loanStats = [0, 0]
loanResults = [[0, 0], [0, 0]]
#           "married","divorced","single"
contactStats = [0, 0, 0]
contactResults = [[0, 0], [0, 0], [0, 0]]
#           yes, no (high or low)
dayStats = [0, 0]
dayResults = [[0, 0], [0, 0]]
#           yes, no (high or low)
monthStats = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
monthResults = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
#           yes, no (high or low)
durationStats = [0, 0]
durationResults = [[0, 0], [0, 0]]
#           yes, no (high or low)
campaignStats = [0, 0]
campaignResults = [[0, 0], [0, 0]]
#           yes, no (high or low)
pdaysStats = [0, 0]
pdaysResults = [[0, 0], [0, 0]]
#           yes, no (high or low)
previousStats = [0, 0]
previousResults = [[0, 0], [0, 0]]
#           "unknown","secondary","primary","tertiary"
poutcomeStats = [0, 0, 0, 0]
poutcomeResults = [[0, 0], [0, 0], [0, 0], [0, 0]]

#           yes, no
labelsStats = [0, 0]



#Pre-Process Numeric Indices
#Gets the Median from each numeric index set and stores it in a temporary array. This array is important as it allows the classifier to compare numerics
#against the median to determine the associated label and type.
processSamples = trainSamples
ageList = []
balanceList = []
dayList = []
durationList = []
campaignList = []
pdaysList = []
previousList = []
for sample in processSamples:

    label = 0

    ageList.append(int(sample[0]))
    balanceList.append(int(sample[5]))
    dayList.append(int(sample[9]))
    durationList.append(int(sample[11]))
    campaignList.append(int(sample[12]))
    pdaysList.append(int(sample[13]))
    previousList.append(int(sample[14]))

ageMedian = statistics.median(ageList)
balanceMedian = statistics.median(balanceList)
dayMedian = statistics.median(dayList)
durationMedian = statistics.median(durationList)
campaignMedian = statistics.median(campaignList)
pdaysMedian = statistics.median(pdaysList)
previousMedian = statistics.median(previousList)


#Sumarize Statistics
#reads off each line of samples and records the information that's needed
for sample in trainSamples:

    label = 0

    #Slot 0 is a yes vote, 1 is a no vote. Bit confusing, but is how it was setup.
    if sample[16] == 'yes':
        labelsStats[0] += 1
        label = 0
    elif sample[16] == 'no':
        labelsStats[1] += 1
        label = 1


    if int(sample[0]) > ageMedian:
        ageStats[0] += 1
        ageResults[0][label] += 1
    elif int(sample[0]) <= ageMedian:
        ageStats[1] += 1
        ageResults[1][label] += 1


    if sample[1] == attribParams[1][0]:
        jobStats[0] += 1
        jobResults[0][label] += 1
    elif sample[1] == attribParams[1][1]:
        jobStats[1] += 1
        jobResults[1][label] += 1
    elif sample[1] == attribParams[1][2]:
        jobStats[2] += 1
        jobResults[2][label] += 1
    elif sample[1] == attribParams[1][3]:
        jobStats[3] += 1
        jobResults[3][label] += 1
    elif sample[1] == attribParams[1][4]:
        jobStats[4] += 1
        jobResults[4][label] += 1
    elif sample[1] == attribParams[1][5]:
        jobStats[5] += 1
        jobResults[5][label] += 1
    elif sample[1] == attribParams[1][6]:
        jobStats[6] += 1
        jobResults[6][label] += 1
    elif sample[1] == attribParams[1][7]:
        jobStats[7] += 1
        jobResults[7][label] += 1
    elif sample[1] == attribParams[1][8]:
        jobStats[8] += 1
        jobResults[8][label] += 1
    elif sample[1] == attribParams[1][9]:
        jobStats[9] += 1
        jobResults[9][label] += 1
    elif sample[1] == attribParams[1][10]:
        jobStats[10] += 1
        jobResults[10][label] += 1
    elif sample[1] == attribParams[1][11]:
        jobStats[11] += 1
        jobResults[11][label] += 1

    
    if sample[2] == attribParams[2][0]:
        maritalStats[0] += 1
        maritalResults[0][label] += 1
    elif sample[2] == attribParams[2][1]:
        maritalStats[1] += 1
        maritalResults[1][label] += 1
    elif sample[2] == attribParams[2][2]:
        maritalStats[2] += 1
        maritalResults[2][label] += 1

    if sample[3] == attribParams[3][0]:
        educationStats[0] += 1
        educationResults[0][label] += 1
    elif sample[3] == attribParams[3][1]:
        educationStats[1] += 1
        educationResults[1][label] += 1
    elif sample[3] == attribParams[3][2]:
        educationStats[2] += 1
        educationResults[2][label] += 1
    elif sample[3] == attribParams[3][3]:
        educationStats[3] += 1
        educationResults[3][label] += 1


    if sample[4] == attribParams[4][0]:
        defaultStats[0] += 1
        defaultResults[0][label] += 1
    elif sample[4] == attribParams[4][1]:
        defaultStats[1] += 1
        defaultResults[1][label] += 1


    if int(sample[5]) > balanceMedian:
        balanceStats[0] += 1
        balanceResults[0][label] += 1
    elif int(sample[5]) <= balanceMedian:
        balanceStats[1] += 1
        balanceResults[1][label] += 1


    if sample[6] == attribParams[6][0]:
        housingStats[0] += 1
        housingResults[0][label] += 1
    elif sample[6] == attribParams[6][1]:
        housingStats[1] += 1
        housingResults[1][label] += 1


    if sample[7] == attribParams[7][0]:
        loanStats[0] += 1
        loanResults[0][label] += 1
    elif sample[7] == attribParams[7][1]:
        loanStats[1] += 1
        loanResults[1][label] += 1

        
    if sample[8] == attribParams[8][0]:
        contactStats[0] += 1
        contactResults[0][label] += 1
    elif sample[8] == attribParams[8][1]:
        contactStats[1] += 1
        contactResults[1][label] += 1
    elif sample[8] == attribParams[8][2]:
        contactStats[2] += 1
        contactResults[2][label] += 1


    if int(sample[9]) > dayMedian:
        dayStats[0] += 1
        dayResults[0][label] += 1
    elif int(sample[9]) <= dayMedian:
        dayStats[1] += 1
        dayResults[1][label] += 1


    if sample[10] == attribParams[10][0]:
        monthStats[0] += 1
        monthResults[0][label] += 1
    elif sample[10] == attribParams[10][1]:
        monthStats[1] += 1
        monthResults[1][label] += 1
    elif sample[10] == attribParams[10][2]:
        monthStats[2] += 1
        monthResults[2][label] += 1
    elif sample[10] == attribParams[10][3]:
        monthStats[3] += 1
        monthResults[3][label] += 1
    elif sample[10] == attribParams[10][4]:
        monthStats[4] += 1
        monthResults[4][label] += 1
    elif sample[10] == attribParams[10][5]:
        monthStats[5] += 1
        monthResults[5][label] += 1
    elif sample[10] == attribParams[10][6]:
        monthStats[6] += 1
        monthResults[6][label] += 1
    elif sample[10] == attribParams[10][7]:
        monthStats[7] += 1
        monthResults[7][label] += 1
    elif sample[10] == attribParams[10][8]:
        monthStats[8] += 1
        monthResults[8][label] += 1
    elif sample[10] == attribParams[10][9]:
        monthStats[9] += 1
        monthResults[9][label] += 1
    elif sample[10] == attribParams[10][10]:
        monthStats[10] += 1
        monthResults[10][label] += 1
    elif sample[10] == attribParams[10][11]:
        monthStats[11] += 1
        monthResults[11][label] += 1


    if int(sample[11]) > durationMedian:
        durationStats[0] += 1
        durationResults[0][label] += 1
    elif int(sample[11]) <= durationMedian:
        durationStats[1] += 1
        durationResults[1][label] += 1


    if int(sample[12]) > campaignMedian:
        campaignStats[0] += 1
        campaignResults[0][label] += 1
    elif int(sample[12]) <= campaignMedian:
        campaignStats[1] += 1
        campaignResults[1][label] += 1


    if int(sample[13]) > pdaysMedian:
        pdaysStats[0] += 1
        pdaysResults[0][label] += 1
    elif int(sample[13]) <= pdaysMedian:
        pdaysStats[1] += 1
        pdaysResults[1][label] += 1


    if int(sample[14]) > previousMedian:
        previousStats[0] += 1
        previousResults[0][label] += 1
    elif int(sample[14]) <= previousMedian:
        previousStats[1] += 1
        previousResults[1][label] += 1


    if sample[15] == attribParams[15][0]:
        poutcomeStats[0] += 1
        poutcomeResults[0][label] += 1
    elif sample[15] == attribParams[15][1]:
        poutcomeStats[1] += 1
        poutcomeResults[1][label] += 1
    elif sample[15] == attribParams[15][2]:
        poutcomeStats[2] += 1
        poutcomeResults[2][label] += 1
    elif sample[15] == attribParams[15][3]:
        poutcomeStats[3] += 1
        poutcomeResults[3][label] += 1



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


print('Age:', ageStats)
print('Job:', jobStats)
print('Marital:', maritalStats)
print('Education:', educationStats)
print('Default:', defaultStats)
print('Balance:', balanceStats)
print('Housing:', housingStats)
print('Loan:', loanResults)
print('Contact:', contactResults)
print('Day:', dayResults)
print('Month:', monthResults)
print('Duration:', durationResults)
print('Campaign:', campaignResults)
print('pDays:', pdaysResults)
print('Previous:', previousResults)
print('pOutCome:', poutcomeResults)


ageGain = calculateInformationGain(ageStats, ageResults)
jobGain = calculateInformationGain(jobStats, jobResults)
maritalGain = calculateInformationGain(maritalStats, maritalResults)
educationGain = calculateInformationGain(educationStats, educationResults)
defaultGain = calculateInformationGain(defaultStats, defaultResults)
balanceGain = calculateInformationGain(balanceStats, balanceResults)
housingGain = calculateInformationGain(housingStats, housingResults)
loanGain = calculateInformationGain(loanStats, loanResults)
contactGain = calculateInformationGain(contactStats, contactResults)
dayGain = calculateInformationGain(dayStats, dayResults)
monthGain = calculateInformationGain(monthStats, monthResults)
durationGain = calculateInformationGain(durationStats, durationResults)
campaignGain = calculateInformationGain(campaignStats, campaignResults)
pdaysGain = calculateInformationGain(pdaysStats, pdaysResults)
previousGain = calculateInformationGain(previousStats, previousResults)
poutcomeGain = calculateInformationGain(poutcomeStats, poutcomeResults)

print('Starting Entropy', baseEntropy())
print('Age Gain:', ageGain)
print('Job Gain:', jobGain)
print('Marital Gain:', maritalGain)
print('Education Gain:', educationGain)
print('Default Gain:', defaultGain)
print('Balance Gain:', balanceGain)
print('Housing Gain:', housingGain)
print('Loan Gain:', loanGain)
print('Contact Gain:', contactGain)
print('Day Gain:', dayGain)
print('Month Gain:', monthGain)
print('Duration Gain:', durationGain)
print('Campaign Gain:', campaignGain)
print('pDays Gain:', pdaysGain)
print('Previous Gain:', previousGain)
print('pOutCome Gain:', poutcomeGain)

#Duration Gain is the max information gain with .059 gain.

# --- Decision Stumps ---

#These mappings allow us to create decision stumps easily
allStatsArray = [ageStats, jobStats, maritalStats, educationStats, defaultStats, balanceStats, housingStats, loanStats, contactStats, dayStats, monthStats, durationStats, campaignStats, pdaysStats, previousStats, poutcomeStats]
allResultsArray = [ageResults, jobResults, maritalResults, educationResults, defaultResults, balanceResults, housingResults, loanResults, contactResults, dayResults, monthResults, durationResults, campaignResults, pdaysResults, previousResults, poutcomeResults]


#Generate Stumps without AdaBoost.
iteration = 0

decisionStumpOutcomes = []

for attribute in attributes:
    #Get the data related to this attribute
    attributeStats = allStatsArray[iteration]
    attributeResults = allResultsArray[iteration]
    attributeParameters = attribParams[iteration]


    parameterVotes = []

    #Go through every possible outcome for an attribute, then generate the decision stump based on it.
    for outcomePair in attributeResults:
        yesVotes = outcomePair[0]
        noVotes = outcomePair[1]

        if(yesVotes > noVotes):
            parameterVotes.append('yes')
        else:
            parameterVotes.append('no')

    decisionStumpOutcomes.append(parameterVotes)

    iteration += 1


print('Age Stump Results:', decisionStumpOutcomes[0])
print('Job Stump Results:', decisionStumpOutcomes[1])
print('Marital Stump Results:', decisionStumpOutcomes[2])
print('Education Stump Results:', decisionStumpOutcomes[3])
print('Default Stump Results:', decisionStumpOutcomes[4])
print('Balance Stump Results:', decisionStumpOutcomes[5])
print('Housing Stump Results:', decisionStumpOutcomes[6])
print('Loan Stump Results:', decisionStumpOutcomes[7])
print('Contact Stump Results:', decisionStumpOutcomes[8])
print('Day Stump Results:', decisionStumpOutcomes[9])
print('Month Stump Results:', decisionStumpOutcomes[10])
print('Duration Stump Results:', decisionStumpOutcomes[11])
print('Campaign Stump Results:', decisionStumpOutcomes[12])
print('pDays Stump Results:', decisionStumpOutcomes[13])
print('Previous Stump Results:', decisionStumpOutcomes[14])
print('pOutCome Stump Results:', decisionStumpOutcomes[15])


# --- AdaBoost ---
weightArray = [ 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16, 1/16] #Start with equally important weights.
stumpAccuracy = [ageGain, jobGain, maritalGain, educationGain, defaultGain, balanceGain, housingGain, loanGain, contactGain, dayGain, monthGain, durationGain, campaignGain, pdaysGain, previousGain, poutcomeGain]


# Step 1 - Get the information gain of all stumps to determine the ordering.
interation = 0
for decisionStump in decisionStumpOutcomes:
    stumpStats = allStatsArray[iteration] # The sets of decisions in the training samples.
    stumpParameters = attribParams[iteration]

    stumpResults = []

    paramCounter = 0
    for parameter in stumpParameters:
        # add in an array of [correct, incorrect]
        stumpResults.append([0, 0])

        #stumpGuess = decisionStump

        stumpResults[paramCounter][0] += 1 #Add one to correct vote
        stumpResults[paramCounter][1] += 1 #Add one to incorrect vote

    iteration += 1


# Step 1 - Gather Statistics on the accuracy of our current stump, using the weights, to find the current iterations accuracy with this stump.


# Step 2 - Calculate the new weights









#buyME = calculateMajorityError(buyingStats, buyingResults)
#maintME = calculateMajorityError(maintStats, maintResults)
#DoorsME = calculateMajorityError(doorsStats, doorsResults)
#PersonsME = calculateMajorityError(personsStats, personsResults)
#LugBootsME = calculateMajorityError(lug_bootStats, lug_bootResults)
#SafetyME = calculateMajorityError(safetyStats, safetyResults)

#print('Buying ME:', buyME)
#print('Maint ME:', maintME)
#print('Doors ME:', DoorsME)
#print('Persons ME:', PersonsME)
#print('LugBoots ME:', LugBootsME)
#print('Safety:ME', SafetyME)
