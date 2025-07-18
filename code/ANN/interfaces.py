from numpy.typing import NDArray
import numpy as np
import abc

class IClassifyDigits(abc.ABC):
    @abc.abstractmethod
    def __call__(self, images: NDArray) -> NDArray[np.int_]:
        """
        Parameters:
            images: np.ndarray of shape (N, 28, 28)
        Returns:
            np.ndarray of shape (N,) with predicted class labels
        """
        ...