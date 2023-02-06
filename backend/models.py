from typing import Any, Optional, Union
from pydantic import BaseModel #Library to validate settings and req/res for FastAPI


#Class for objects used when creating and training a model
class TrainModel(BaseModel):
    model_name: str
    hyperparams: dict[str, Any]
    epochs: int


#Class for objects used fetching pretrained models
class PredictModel(BaseModel):
    input_image: Any
    model_name: str


#Class for objects used when deleting pretrained models
class DeleteModel(BaseModel):
    model_name: str
    model_version: Optional[Union[list[int], int]]  # list | int in python 10
