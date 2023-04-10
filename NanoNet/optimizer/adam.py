from .rms_promp import RMSPromp
from .sgd_momentum import SGD_Momentum

class ADAM(SGD_Momentum, RMSPromp):
    pass