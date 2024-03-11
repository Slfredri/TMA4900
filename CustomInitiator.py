from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.types.update import Update
from stonesoup.types.hypothesis import SingleProbabilityHypothesis
from stonesoup.types.state import GaussianState
import numpy as np

class RestrictedInitiator(MultiMeasurementInitiator):
    """
    Initiator for restricted field of view.

    This initiator restricts object initiation to the immediate neighborhood on
    the two edges of the field of view.
    """
    

    def __init__(self, *args, FOV=[0, 400], range=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.FOV = FOV
        self.range = range
        self.prior_state = None
        

    def initiate(self, detections, timestamp, **kwargs):

        # Calculate initial state based on measurements falling within FOV and range
        initial_position, initial_velocity = self.calculate_initial_state(detections)

        # Set the prior state based on the calculated initial position and velocity
        self.prior_state = GaussianState([[initial_position], [initial_velocity]], np.diag([10,2]))  # Adjust covariance as needed
        self.initiator = SimpleMeasurementInitiator(
            prior_state=self.prior_state,
            measurement_model=self.measurement_model
        )
        sure_tracks = set()
        associated_detections = set()

        if self.holding_tracks:
            associations = self.data_associator.associate(
                self.holding_tracks, detections, timestamp)

            
        
            for track, hypothesis in associations.items():
                if hypothesis:
                    state_post = self.updater.update(hypothesis)
                    track.append(state_post)
                    associated_detections.add(hypothesis.measurement)
                else:
                    track.append(hypothesis.prediction)

                if sum(1 for state in track if not self.updates_only or isinstance(state, Update)) \
                        >= self.min_points:
                    sure_tracks.add(track)
                    self.holding_tracks.remove(track)

            self.holding_tracks -= self.deleter.delete_tracks(self.holding_tracks)



        # Filter out detections outside the initiation range
        filtered_detections = set()
        for detection in detections - associated_detections:
            if self.FOV[0] <= detection.state_vector <= self.FOV[0]+self.range or self.FOV[1]-self.range <= detection.state_vector <= self.FOV[1]:
                filtered_detections.add(detection)

        self.holding_tracks |= self.initiator.initiate(
            filtered_detections, timestamp)

        return sure_tracks


    def calculate_initial_state(self, detections):
        # Calculate initial position and velocity based on measurements
        valid_measurements_left = [detection.state_vector for detection in detections if self.FOV[0] <= detection.state_vector <= self.FOV[0] + self.range]
        valid_measurements_right = [detection.state_vector for detection in detections if self.FOV[1] - self.range <= detection.state_vector <= self.FOV[1]]
        
        if not valid_measurements_left and not valid_measurements_right:
            # If no valid measurements, default to FOV borders
            initial_position = np.random.choice([self.FOV[0], self.FOV[1]])
            initial_velocity = 10 if initial_position == self.FOV[0] else -10
        elif valid_measurements_left:
            # Use the average of valid measurements as initial position
            initial_position = np.mean(valid_measurements_left)
            initial_velocity = 10
        elif valid_measurements_right:
            initial_position = np.mean(valid_measurements_right)
            initial_velocity = -10

        return initial_position, initial_velocity

