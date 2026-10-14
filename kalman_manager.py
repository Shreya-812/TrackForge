import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from kalman_tracker import KalmanTracker

classVariance = {
    0: 70,
    1: 80,
    2: 110,
    3: 120,
    5: 100,
    7: 100,
}

class KalmanManager:
    def __init__(self, measureVariance, distThreshold, coastThreshold):
        self.trackers = []
        self.measureVariance = measureVariance
        self.distThreshold = distThreshold
        self.coastThreshold = coastThreshold
        self.IDcounter = 0

    def updateFrame(self, detections, objList):
        #case 1 no existing trackers ie first frame or long periods of video with no new detections
        if len(self.trackers) == 0:
            for d, v in zip(detections, objList):
                newTracker = KalmanTracker(d, classVariance[v], self.measureVariance, self.IDcounter)
                self.trackers.append(newTracker)
                self.IDcounter += 1

        #case 2 no new detections
        elif len(detections) == 0:
            for t in self.trackers:
                t.predict()
            self.trackers = [t for t in self.trackers if t.numPredicts <= self.coastThreshold]

        #case 3 normal detections and tracking
        else:
            # 1. predict
            for t in self.trackers:
                t.predict()

            # 2. build cost matrix
            tPositions = np.array([t.getCoord() for t in self.trackers])
            dPositions = np.array([d for d in detections])
            costMatrix = cdist(tPositions, dPositions, 'euclidean')

            # 3. gate invalid associations
            gatedCostMatrix = costMatrix.copy()
            gatedCostMatrix[gatedCostMatrix > self.distThreshold] = 1e6

            # 4. hungarian match
            row, col = linear_sum_assignment(gatedCostMatrix)

            # 5. threshold check distance for all
            unmatchDet = []
            for r, c in zip(row, col):
                if costMatrix[r, c].sum() < self.distThreshold:
                    self.trackers[r].update(detections[c])
                else:
                    unmatchDet.append(c)

            #5. set difference to find unmatched detections → create new trackers
            matchDet = set(col)
            allDet = set(range(len(detections)))
            noMatchDet = allDet - matchDet
            noMatchDet = noMatchDet | set(unmatchDet)
            for d in noMatchDet:
                newTracker = KalmanTracker(detections[d], classVariance[objList[d]], self.measureVariance, self.IDcounter)
                self.trackers.append(newTracker)
                self.IDcounter += 1

            #6. filter self.trackers by coastThreshold
            self.trackers = [t for t in self.trackers if t.numPredicts <= self.coastThreshold]

