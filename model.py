from typing import Union
from typing_extensions import Self, Callable

import torch
from torch import nn

from normalization import Normalization


class Model(nn.Module):

	def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, normalization: Callable[tuple[int], Normalization], \
			softmax: bool = False) -> Self:
		super(Model, self).__init__()
		self.__hidden_size = hidden_size
		self.__input_size = input_size
		self.__num_layers = num_layers
		self.__output_size = output_size
		self.__softmax = softmax
		self._normalization = normalization((self.__input_size,))
		self.__lstm = nn.LSTM(self.__input_size, self.__hidden_size, self.__num_layers, batch_first=True)
		self.__linear = nn.Linear(self.__hidden_size, self.__output_size)

	def forward(self, x: Union[torch.FloatTensor, torch.cuda.FloatTensor], lstm_state: tuple[Union[torch.FloatTensor, torch.cuda.FloatTensor]]) \
			-> tuple[Union[torch.FloatTensor, torch.cuda.FloatTensor]]:
		if self.training:
			x = self._normalization.normalize(x, doUpdate=True)
		else:
			x = self._normalization.normalize(x, doUpdate=False)
		output, (lstm_hidden_state, lstm_c_state) = self.__lstm(x, lstm_state)
		output = self.__linear(output)
		if self.__softmax:
			output = nn.functional.softmax(output, dim=-1)
		return (output, lstm_hidden_state, lstm_c_state)

	def to(self, *args: any, **kwargs: dict[str, any]) -> None:
		super(Model, self).to(*args, **kwargs)
		self._normalization.to(*args, **kwargs)
