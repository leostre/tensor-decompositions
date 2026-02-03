from typing import Any, Union


type TensorLike = Any
'''Tensorly supports work with different tensor backends (numpy, torch.tensor and so on), 
but it doesnt describe abstract class for it. 
So the tensor can be of `Any` type depending on backend setted in `tl.set_backend` .'''

Number = Union[int, float]
'''Type, widely used for 'rank' typing'''