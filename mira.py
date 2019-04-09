# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = True
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.001, 0.002, 0.004]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """

        if len(validationLabels) > 0:
            Cgrid = [0.0007] + Cgrid

        bestC = 0
        mostCorrect = -1
        bestWeights = util.Counter()

        initWeights = self.weights.copy()

        for C in Cgrid:

            print("")
            print("C : " + str(C))

            # Reset weights
            self.weights = initWeights.copy()

            # Train
            for iter in range(self.max_iterations):
                print("Iter " + str(iter))
                for i in range(len(trainingData)):

                    observation = trainingData[i]

                    labelGuess = self.classify([observation])[0]

                    labelCorrect = trainingLabels[i]
                    if labelGuess != labelCorrect:

                        wGuess = self.weights[labelGuess]
                        wCorrect = self.weights[labelCorrect]
                        denom = 2.0 * (observation * observation)
                        frac = ((wGuess - wCorrect) * observation + 1.0) / denom

                        tau = min(C, frac)

                        observation.divideAll(1.0 / tau)
                        self.weights[labelGuess] -= observation
                        self.weights[labelCorrect] += observation

            # Validation
            numCorrect = 0
            labelGuesses = self.classify(validationData)
            for i in range(len(validationLabels)):
                if labelGuesses[i] == validationLabels[i]:
                    numCorrect += 1

            # If best performance so far, store weights
            if numCorrect > mostCorrect:
                bestC = C
                mostCorrect = numCorrect
                bestWeights = self.weights

        print("Best C : " + str(bestC))

        # Store the best weights
        self.weights = bestWeights

        if len(validationData) > 0:
            self.trainAndTune(trainingData + validationData, trainingLabels + validationLabels, [], [], [bestC])


    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


