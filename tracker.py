from datetime import timedelta
import numpy as np
from stonesoup.functions import gm_reduce_single
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.array import StateVectors
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.types.prediction import GaussianStatePrediction
from utils import count_points_between
from utils import map_point_between_ranges


class JPDAtracker:
    def __init__(self, data_associator, updater, deleter, initiator):
        self.data_associator = data_associator
        self.updater = updater
        self.deleter = deleter
        self.initiator = initiator
        self.track_probabilities = {}
        self.all_probabilities = {}
        self.all_probabilities_timestamps = {}
        
    

    def track(self, all_measurements, start_time, dt = 1, rms_processed=0, phi_1=0, phi_2=0, pi_0_1=0, pi_0_2=0):
        tracks, all_tracks = set(), set()

        for n, measurements in enumerate(all_measurements):
            
            current_time = start_time + timedelta(seconds=n * dt)

            associations = self.data_associator.associate(tracks, measurements, current_time)

            unassociated_detections = set(measurements)
            for track, multihypothesis in associations.items():
                
                posterior_states = []
                posterior_state_weights = []
                amplitudes = []
                for hypothesis in multihypothesis:
                    if not hypothesis:
                        posterior_states.append(hypothesis.prediction)
                        amplitudes.append(hypothesis.measurement.metadata.get('amplitude'))
                    else:
                        posterior_states.append(self.updater.update(hypothesis))
                        amplitudes.append(hypothesis.measurement.metadata.get('amplitude'))
                       
                        

                    posterior_state_weights.append(hypothesis.probability)
                
                means = StateVectors([state.state_vector for state in posterior_states])
                covars = np.stack([state.covar for state in posterior_states], axis=2)
                weights = np.asarray(posterior_state_weights)
                
            
                post_mean, post_covar = gm_reduce_single(means, covars, weights)
                
                track_id = track.id
                
                missed_detection_weight = next(hyp.weight for hyp in multihypothesis if not hyp)

                # Check if at least one reasonable measurement...
                if any(hypothesis.weight > missed_detection_weight
                        for hypothesis in multihypothesis):
                    # ...and if so use update type
                    track.append(GaussianStateUpdate(
                        post_mean, post_covar,
                        multihypothesis,
                        multihypothesis[0].measurement.timestamp))

                    lik_1 = np.sum([phi_1.pdf(amp) * float(weight) if amp != None else 0 for amp, weight in zip(amplitudes,posterior_state_weights)])
                    lik_2 = np.sum([phi_2.pdf(amp) * float(weight) if amp != None else 0 for amp, weight in zip(amplitudes,posterior_state_weights)])
                    
                   

                    if track_id in self.track_probabilities:
                        
                        pi_tm_1 = self.track_probabilities[track_id]['car_probabilities'][-1]
                        pi_tm_2 = self.track_probabilities[track_id]['train_probabilities'][-1]
                    else:
                        
                        pi_tm_1 = pi_0_1
                        pi_tm_2 = pi_0_2
        
                    pi_t_1 = lik_1 * pi_tm_1
                    pi_t_2 = lik_2 * pi_tm_2
                    # normalize
                    pi_t_1_norm = pi_t_1 / (pi_t_1 + pi_t_2)
                    pi_t_2_norm = pi_t_2 / (pi_t_1 + pi_t_2)

                    pi_t_1 = pi_t_1_norm
                    pi_t_2 = pi_t_2_norm

                else:
                    # ...and if not, treat as a prediction
                    track.append(GaussianStatePrediction(
                        post_mean, post_covar,
                        multihypothesis[0].prediction.timestamp))
                    
                    if track_id in self.track_probabilities:
                        
                        pi_t_1 = self.track_probabilities[track_id]['car_probabilities'][-1]
                        pi_t_2 = self.track_probabilities[track_id]['train_probabilities'][-1]
                    else:
                        pi_t_1 = pi_0_1
                        pi_t_2 = pi_0_2

                self.track_probabilities[track_id]['car_probabilities'].append(pi_t_1)
                self.track_probabilities[track_id]['train_probabilities'].append(pi_t_2)
                
                if track_id not in self.all_probabilities:
                    self.all_probabilities[track_id] = {
                        'car_probabilities': [pi_t_1],
                        'train_probabilities': [pi_t_2]
                    }
                else: 
                    self.all_probabilities[track_id]['car_probabilities'].append(pi_t_1)
                    self.all_probabilities[track_id]['train_probabilities'].append(pi_t_2)
            


                if track_id not in self.all_probabilities_timestamps:
                    self.all_probabilities_timestamps[track_id] = [multihypothesis[0].prediction.timestamp]
                    
                else: 
                    self.all_probabilities_timestamps[track_id].append(multihypothesis[0].prediction.timestamp)
                    

                # any detections in multihypothesis that had an
                # association score (weight) lower than or equal to the
                # association score of "MissedDetection" is considered
                # unassociated - candidate for initiating a new Track
                for hyp in multihypothesis:
                    if hyp.weight > missed_detection_weight:
                        if hyp.measurement in unassociated_detections:
                            unassociated_detections.remove(hyp.measurement)

            ######################################################################################
            # Carry out deletion and initiation
            tracks -= self.deleter.delete_tracks(tracks)
            
            # Delete obsolete track probabilities
            for trackid in set(self.track_probabilities) - {track.id for track in tracks}:
                del self.track_probabilities[trackid]

            tracks |= self.initiator.initiate(unassociated_detections,
                                              current_time)

            for trackid in {track.id for track in tracks}:
                if trackid not in self.track_probabilities:
                    self.track_probabilities[trackid] = {
                        'car_probabilities': [pi_0_1],
                        'train_probabilities': [pi_0_2]
                    }
            all_tracks |= tracks


        return all_tracks



from utils import map_point_between_ranges
from stonesoup.types.detection import Detection
from tqdm import tqdm



def create_measurements(times, rms_processed, measurement_model, dx, chmin_rel, chmax_rel):
    all_measurements = []
    all_amplitudes = []
    for timestamp, rms_values in zip(times, rms_processed):
        measurement_set = set()
        amplitude_set = set()
        for value in rms_values:
            if value != 0:
                
                measurement = Detection(
                    timestamp=timestamp,
                    state_vector=map_point_between_ranges(np.array([np.where(rms_values == value)[0]]),source_range=(0,rms_processed.shape[1]), target_range=(dx*chmin_rel, dx*chmax_rel)),
                    measurement_model=measurement_model,
                    metadata =  {'amplitude': value}
                )
                measurement_set.add(measurement)

        all_measurements.append(measurement_set)
    return all_measurements







    """
class JPDAtracker:
    def __init__(self, data_associator, updater, deleter, initiator, random_seed):
        self.data_associator = data_associator
        self.updater = updater
        self.deleter = deleter
        self.initiator = initiator
        if random_seed is not None:
            np.random.seed(random_seed)  # Set the random seed if provided

    def track(self, all_measurements, start_time, dt = 1, rms_processed=0, phi_1=0, phi_2=0, pi_0_1=0, pi_0_2=0):
        tracks, all_tracks = set(), set()

        for n, measurements in enumerate(all_measurements):

            current_time = start_time + timedelta(seconds=n * dt)

            associations = self.data_associator.associate(tracks, measurements, current_time)

            unassociated_detections = set(measurements)
            for track, multihypothesis in associations.items():
                
                posterior_states = []
                posterior_state_weights = []
                amplitudes = []
                for hypothesis in multihypothesis:
                    if not hypothesis:
                        posterior_states.append(hypothesis.prediction)
                        amplitudes.append(0)
                    else:
                        posterior_states.append(self.updater.update(hypothesis))
                        #print([int(count_points_between(start_time,timedelta(seconds=dt),hypothesis.measurement.timestamp))-1,int(map_point_between_ranges(hypothesis.measurement.state_vector[0],reverse=True))])
                        amplitudes.append(rms_processed[int(count_points_between(start_time,timedelta(seconds=dt),hypothesis.measurement.timestamp))-1,int(map_point_between_ranges(hypothesis.measurement.state_vector[0],reverse=True))])
                        

                    posterior_state_weights.append(hypothesis.probability)
                
                means = StateVectors([state.state_vector for state in posterior_states])
                covars = np.stack([state.covar for state in posterior_states], axis=2)
                weights = np.asarray(posterior_state_weights)
                
                lik_1 = np.array([phi_1.pdf(amp) for amp in amplitudes])
                lik_2 = np.array([phi_2.pdf(amp) for amp in amplitudes])
                
                print(f"Amplitudes: {}")
                print(f"means: {means}")
                print(f"weights: {weights}")
                post_mean, post_covar = gm_reduce_single(means, covars, weights)
                post_amplitudes = weights.T @ amplitudes
                print(f"post_amplitudes: {post_amplitudes}")

                missed_detection_weight = next(hyp.weight for hyp in multihypothesis if not hyp)

                # Check if at least one reasonable measurement...
                if any(hypothesis.weight > missed_detection_weight
                        for hypothesis in multihypothesis):
                    # ...and if so use update type
                    track.append(GaussianStateUpdate(
                        post_mean, post_covar,
                        multihypothesis,
                        multihypothesis[0].measurement.timestamp))
                else:
                    # ...and if not, treat as a prediction
                    track.append(GaussianStatePrediction(
                        post_mean, post_covar,
                        multihypothesis[0].prediction.timestamp))

                # any detections in multihypothesis that had an
                # association score (weight) lower than or equal to the
                # association score of "MissedDetection" is considered
                # unassociated - candidate for initiating a new Track
                for hyp in multihypothesis:
                    if hyp.weight > missed_detection_weight:
                        if hyp.measurement in unassociated_detections:
                            unassociated_detections.remove(hyp.measurement)

            # Carry out deletion and initiation
            tracks -= self.deleter.delete_tracks(tracks)
            
            tracks |= self.initiator.initiate(unassociated_detections,
                                              current_time)
            all_tracks |= tracks

        return all_tracks
    """





