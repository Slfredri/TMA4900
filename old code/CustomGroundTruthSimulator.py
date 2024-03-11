from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.base import Property
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
import numpy as np
from ordered_set import OrderedSet


class CustomMultiTargetGroundTruthSimulator(MultiTargetGroundTruthSimulator):
    fov: tuple = Property(default=(0, 400), doc="Field of view as a tuple (start, end)")
    initial_velocity: float = Property(default=5.0, doc="Initial velocity for objects")

    def _new_target(self, time, random_state):
        # Initialize objects at either end of the FOV with the given initial velocity
        if random_state.rand() < 0.5:
            position = self.fov[0]
            velocity = self.initial_velocity
        else:
            position = self.fov[1]
            velocity = -self.initial_velocity

        state_vector = np.array([position, velocity])

        gttrack = GroundTruthPath()
        gttrack.append(GroundTruthState(
            state_vector=state_vector,
            timestamp=time,
            metadata={"index": self.index})
        )
        return gttrack

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self, random_state=None):
        time = self.initial_state.timestamp or datetime.datetime.now()
        random_state = random_state if random_state is not None else self.random_state
        number_steps_remaining = self.number_steps

        if self.preexisting_states or self.initial_number_targets:
            # Use preexisting_states to make some groundtruth paths
            preexisting_paths = OrderedSet(
                self._new_target(time, random_state) for _ in range(self.initial_number_targets))

            # Union the two sets
            groundtruth_paths = preexisting_paths

            number_steps_remaining -= 1
            yield time, groundtruth_paths
            time += self.timestep

        else:
            groundtruth_paths = OrderedSet()

        for _ in range(number_steps_remaining):
            # Random drop tracks
            groundtruth_paths.difference_update(
                gttrack
                for gttrack in groundtruth_paths.copy()
                if random_state.rand() <= self.death_probability)

            # Move tracks forward through the FOV using the transition model
            for gttrack in groundtruth_paths:
                self.index = gttrack[-1].metadata.get("index")
                trans_state_vector = self.transition_model.function(
                    gttrack[-1], noise=True, time_interval=self.timestep)

                # Check if approaching the other side of the FOV
                if (gttrack[-1].state_vector[1] > 0 and trans_state_vector[0] > self.fov[1]) or \
                        (gttrack[-1].state_vector[1] < 0 and trans_state_vector[0] < self.fov[0]):
                    groundtruth_paths.remove(gttrack)  # Remove the track as it's leaving the FOV
                else:
                    gttrack.append(GroundTruthState(
                        state_vector=trans_state_vector, timestamp=time,
                        metadata={"index": self.index}))

            # Random create
            for _ in range(random_state.poisson(self.birth_rate)):
                self.index = 0
                # Initialize new objects at either end of the FOV with the given initial velocity
                if random_state.rand() < 0.5:
                    position = self.fov[0]
                    velocity = self.initial_velocity
                else:
                    position = self.fov[1]
                    velocity = -self.initial_velocity

                state_vector = np.array([position, velocity])

                gttrack = GroundTruthPath()
                gttrack.append(GroundTruthState(
                    state_vector=state_vector, timestamp=time,
                    metadata={"index": self.index})
                )
                groundtruth_paths.add(gttrack)

            yield time, groundtruth_paths
            time += self.timestep

