from typing import Union
from typing_extensions import Self

import torch
from torch import nn

from normalization import RunningNormalization


class Model(nn.Module):

	def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, softmax: bool = False) -> Self:
		super(Model, self).__init__()
		self.__hidden_size = hidden_size
		self.__input_size = input_size
		self.__num_layers = num_layers
		self.__output_size = output_size
		self.__softmax = softmax
		self.__running_normalization = RunningNormalization((self.__input_size))
		self.__lstm = nn.LSTM(self.__input_size, self.__hidden_size, self.__num_layers, batch_firts=True)
		self.__linear = nn.Linear(self.__hidden_size, self.__output_size)

	def forward(self, x: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> tuple[Union[torch.FloatTensor, torch.cuda.FloatTensor]]:
		if self.training:
			x = self.__running_normalization.normalize(x, doUpdate=True)
		else:
			x = self.__running_normalization.normalize(x, doUpdate=False)
		output, (lstm_hidden_state, lstm_c_state) = self.__lstm(x)
		output = self.__linear(output)
		if self.__softmax:
			output = nn.functional.softmax(output)
		return (output, lstm_hidden_state, lstm_c_state)

	def to(self, *args: any, **kwargs: dict[str, any]) -> Self:
		super(Model, self).to(*args, **kwargs)
		self.__running_normalization.mean.to(*args, **kwargs)
		self.__running_normalization.var.to(*args, **kwargs)
		self.__running_normalization.std.to(*args, **kwargs)
		return self