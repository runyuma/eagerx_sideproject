import eagerx
import numpy as np
from eagerx.core.specs import ProcessorSpec


class DecomposedAngle(eagerx.Processor):
    @classmethod
    def make(cls, convert_to: str = "theta_theta_dot") -> ProcessorSpec:
        spec = cls.get_specification()
        return spec

    def initialize(self, spec: ProcessorSpec):
        pass

    def convert(self, msg: np.ndarray) -> np.ndarray:
        alpha1 = msg[0]
        alpha2 = msg[1]
        return np.array([np.cos(alpha1), np.sin(alpha1),np.cos(alpha2), np.sin(alpha2)], dtype="float32")
class DecomposedAngle_vel(eagerx.Processor):
    @classmethod
    def make(cls, convert_to: str = "theta_theta_dot") -> ProcessorSpec:
        spec = cls.get_specification()
        return spec

    def initialize(self, spec: ProcessorSpec):
        pass

    def convert(self, msg: np.ndarray) -> np.ndarray:
        alpha1dot = msg[0]
        alpha2dot = msg[1]
        return np.array([alpha1dot, alpha2dot], dtype="float32")


class ObsWithDecomposedAngle(eagerx.Processor):
    @classmethod
    def make(cls, convert_to: str = "theta_theta_dot") -> ProcessorSpec:
        spec = cls.get_specification()
        spec.config.convert_to = convert_to
        return spec

    def initialize(self, spec: ProcessorSpec):
        self.convert_to = spec.config.convert_to

    def convert(self, msg: np.ndarray) -> np.ndarray:
        if not len(msg):  # No data
            data = np.array(msg, dtype="float32")
        elif self.convert_to == "trig_theta_dot":
            data = np.array([np.sin(-msg.data[0]), np.cos(msg.data[0]), -msg.data[1]], dtype="float32")
        elif self.convert_to == "theta_theta_dot":
            cos_th = msg.data[0]
            sin_th = msg.data[1]
            data = np.array([-np.arctan2(sin_th, cos_th), -msg.data[2]], dtype="float32")
        else:
            raise NotImplementedError(f"Convert_to '{self.convert_to}' not implemented.")
        return data


class Negate(eagerx.Processor):
    @classmethod
    def make(cls) -> ProcessorSpec:
        return cls.get_specification()

    def initialize(self, spec: ProcessorSpec):
        pass

    def convert(self, msg: np.ndarray) -> np.ndarray:
        return -msg


class VoltageToMotorTorque(eagerx.Processor):
    @classmethod
    def make(cls, K: float, R: float) -> ProcessorSpec:
        # Initialize spec with default arguments
        spec = cls.get_specification()
        spec.config.K = K
        spec.config.R = R
        return spec

    def initialize(self, spec: ProcessorSpec):
        self.K = spec.config.K
        self.R = spec.config.R

    def convert(self, msg: np.ndarray) -> np.ndarray:
        return np.array([-msg[0] * self.K / self.R], dtype="float32")
