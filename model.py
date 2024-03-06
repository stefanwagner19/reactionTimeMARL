from typing import Union
from typing_extensions import Self, Callable

import torch
from torch import nn

from normalization import Normalization


class Model(nn.Module):

	def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, normalization: Callable[Tuple[int], Normalization], \
			softmax: bool = False) -> Self:
		super(Model, self).__init__()
		self.__hidden_size = hidden_size
		self.__input_size = input_size
		self.__num_layers = num_layers
		self.__output_size = output_size
		self.__softmax = softmax
		self.__normalization = normalization((self.__input_size))
		self.__lstm = nn.LSTM(self.__input_size, self.__hidden_size, self.__num_layers, batch_firts=True)
		self.__linear = nn.Linear(self.__hidden_size, self.__output_size)

	def forward(self, x: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> tuple[Union[torch.FloatTensor, torch.cuda.FloatTensor]]:
		if self.training:
			x = self.__normalization.normalize(x, doUpdate=True)
		else:
			x = self.__normalization.normalize(x, doUpdate=False)
		output, (lstm_hidden_state, lstm_c_state) = self.__lstm(x)
		output = self.__linear(output)
		if self.__softmax:
			output = nn.functional.softmax(output)
		return (output, lstm_hidden_state, lstm_c_state)

	def to(self, *args: any, **kwargs: dict[str, any]) -> None:
		super(Model, self).to(*args, **kwargs)
		self.__normalization.to(*args, **kwargs)
