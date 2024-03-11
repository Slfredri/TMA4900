from typing import Sequence, List
import numpy as np
from stonesoup.base import Property
from stonesoup.deleter.base import Deleter


class CovarianceBasedDeleter(Deleter):
    """ Track deleter based on covariance matrix size.

    Deletes tracks whose state covariance matrix (more specifically its trace)
    exceeds a given threshold.
    """

    covar_trace_thresh: float = Property(doc="Covariance matrix trace threshold")
    mapping: Sequence[int] = Property(default=None,
                                      doc="Track state vector indices whose corresponding "
                                          "covariances' sum is to be considered. Defaults to"
                                          "None, whereby the entire track covariance trace is "
                                          "considered.")

    def check_for_deletion(self, track, **kwargs):
        """Check if a given track should be deleted

        A track is flagged for deletion if the trace of its state covariance
        matrix is higher than :py:attr:`~covar_trace_thresh`.

        Parameters
        ----------
        track : Track
            A track object to be checked for deletion.

        Returns
        -------
        bool
            `True` if track should be deleted, `False` otherwise.
        """

        diagonals = np.diag(track.state.covar)
        if self.mapping:
            track_covar_trace = np.sum(diagonals[self.mapping])
        else:
            track_covar_trace = np.sum(diagonals)

        if track_covar_trace > self.covar_trace_thresh:
            return True
        return False


class CustomDeleter(CovarianceBasedDeleter):
    """Custom track deleter based on field of view (FOV) and covariance matrix size.

    Deletes tracks if they are outside the specified field of view or if their covariance
    matrix trace exceeds a given threshold.

    Parameters
    ----------
    covar_trace_thresh : float
        Covariance matrix trace threshold.
    mapping : list of int, optional
        Track state vector indices whose corresponding covariances' sum is to be considered.
        Defaults to None, whereby the entire track covariance trace is considered.
    fov : list of float
        Field of view bounds [lower_bound, upper_bound].

    Attributes
    ----------
    covar_trace_thresh : float
        Covariance matrix trace threshold.
    mapping : list of int or None
        Track state vector indices whose corresponding covariances' sum is to be considered.
    fov : list of float
        Field of view bounds [lower_bound, upper_bound].
    """

    fov: List[float] = Property(doc="Field of view bounds [lower_bound, upper_bound]")

    def check_for_deletion(self, track, **kwargs):
        """Check if a given track should be deleted based on the field of view and covariance trace.

        A track is flagged for deletion if its position exceeds the field of view bounds or
        if the trace of its state covariance matrix is higher than the threshold.

        Parameters
        ----------
        track : Track
            A track object to be checked for deletion.

        Returns
        -------
        bool
            `True` if the track is outside the field of view or covariance threshold and should be
            deleted, `False` otherwise.
        """

        # Check covariance trace
        if super().check_for_deletion(track):
            return True

        # Check field of view
        position = track.state_vector[0]  # Assuming position is the first element in the state vector
        if position < self.fov[0] or position > self.fov[1]:
            return True

        return False
