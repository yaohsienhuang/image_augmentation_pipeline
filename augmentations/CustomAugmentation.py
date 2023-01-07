import numpy as np
import random

class RandomBrightness:
    def __init__(self, delta=16):
        assert 0 <= delta <= 255, "brightness must between 0 and 255."
        self.delta = delta

    def __call__(self, image):
        delta = random.uniform(-self.delta, self.delta)
        image = np.clip(image + delta, 0.0, 255.0)
        return image
    
class RandomContrast:
    def __init__(self, lower=0.5, upper=1.5):
        assert upper >= lower, "contrast upper must be >= lower."
        assert lower >= 0, "contrast lower must be non-negative."
        self.lower = lower
        self.upper = upper

    def __call__(self, image):
        alpha = random.uniform(self.lower, self.upper)
        image = np.clip(image * alpha, 0.0, 255.0)
        return image
        
        