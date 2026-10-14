import numpy as np
from collections import deque

class KalmanTracker:
    def __init__(self, state, objVariance, measureVariance, ID):
        self.ID = ID
        self.tails = deque(maxlen=15)
        x, y = state
        self.stateVector = np.array([x, y, 0, 0])
        self.stateVector = self.stateVector.reshape(4,1)
        self.covariance = 100 * np.identity(4)
        self.measureVariance = measureVariance * np.identity(2)
        self.objVariance = objVariance * np.identity(4)
        self.numPredicts = 0
        self.motionModel = np.array(([1, 0, 0.1, 0],
                                     [0, 1, 0, 0.1],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1],))

    def getCoord(self):
        x = self.stateVector[0][0]
        y = self.stateVector[1][0]
        coord = (x, y)
        return coord

    def getID(self):
        return self.ID

    def getTails(self):
        return self.tails

    def predict(self):
        self.tails.append(self.getCoord())
        self.stateVector = self.motionModel @ self.stateVector
        self.covariance = self.motionModel @ self.covariance @ self.motionModel.T + self.objVariance
        self.numPredicts += 1

    def update(self, newMeasure):
        #observation matrix
        H = np.array(([1, 0, 0, 0],
                      [0, 1, 0, 0]))

        '''
        #(2x4) @ (4x1) = (2x1)
        H @ self.stateVector = c
        #(2x1) - (2x1) = (2x1)
        newMeasure - c
        '''
        x, y, = newMeasure
        updateValues = np.array(([x],
                                 [y]))
        residual = updateValues - H @ self.stateVector

        '''
        #(2x4) @ (4x4) = (2x4)
        H @ self.covariance
        #(2x4) @ (4x2) = (2x2)
        self.covariance @ H.T
        #(2x2) + (2x2) = (2x2)
        + self.measureVariance = b

        #(4x4) @ (4x2) = (4x2)
        self.covariance @ H.T = a
        #(4x2) @ (2x2) = (4x2)
        a @ b
        '''

        kalmanGain = self.covariance @ H.T @ np.linalg.inv(H @ self.covariance @ H.T + self.measureVariance)

        #(4x2) @ (2x1) = (4x1)
        #kalmanGain @ residual

        self.stateVector = self.stateVector + (kalmanGain @ residual)
        self.covariance = (np.identity(4) - kalmanGain @ H) @ self.covariance
        self.numPredicts = 0