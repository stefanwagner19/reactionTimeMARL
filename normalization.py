import abc
from typing import Union
from typing_extensions import Self

import torch


class Normalization(abc.ABC):

	def __init__(self, feature_shape: tuple[int]) -> Self:
		self._feature_shape = feature_shape

	def normalize(self, x: Union[torch.FloatTensor, torch.cuda.FloatTensor], doUpdate: bool = False) -> tuple[Union[torch.FloatTensor, torch.cuda.FloatTensor]]:
		if doUpdate:
			self._update(x)
		return self._normalize(x)

	@abc.abstractmethod
	def _normalize(self, x: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> tuple[Union[torch.FloatTensor, torch.cuda.FloatTensor]]:
		pass

	@abc.abstractmethod
	def _update(self, x: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> None:
		pass

	@abc.abstractmethod
	def to(self, *args: any, **kwargs: dict[str, any]) -> None:
		pass


class NoNormalization(Normalization):

	def __init__(self, feature_shape: tuple[int]) -> Self:
		super(NoNormalization, self).__init__(feature_shape)

	def _normalize(self, x: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> tuple[Union[torch.FloatTensor, torch.cuda.FloatTensor]]:
		return x

	def _update(self, x: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> None:
		pass

	def to(self, *args: any, **kwargs: dict[str, any]) -> None:
		pass


class RunningNormalization(Normalization):

	def __init__(self, feature_shape: tuple[int], eps: float = 1e-6) -> Self:
		super(RunningNormalization, self).__init__(feature_shape)
		self.__eps = eps
		self.__count = 0
		self.mean = torch.zeros(self._feature_shape)
		self.var = torch.ones(self._feature_shape)
		self.std = torch.ones(self._feature_shape)

	def _normalize(self, x: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> tuple[Union[torch.FloatTensor, torch.cuda.FloatTensor]]:
		return (x - self.mean) / self.std

	def _update(self, x: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> None:
		'''
		based on Welford's algorithm
		https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
		'''
		x = x.view(-1, *self._feature_shape)
		x_mean = torch.mean(x, dim=0)
		x_var = torch.var(x, dim=0)
		x_len = x.size(0)

		new_count = self.__count + x_len
		delta = x_mean - self.mean
		m2_b = x_var * x_len
		m2_a = self.var * self.__count
		m2 = m2_a + m2_b + delta**2 * self.__count * x_len / (new_count + self.__eps)

		self.mean += delta * x_len / (new_count + self.__eps)
		self.var = m2 / (new_count - 1 + self.__eps)
		self.std = torch.sqrt(self.var)
		self.__count = new_count

	def to(self, *args: any, **kwargs: dict[str, any]) -> None:
		self.mean = self.mean.to(*args, **kwargs)
		self.var = self.var.to(*args, **kwargs)
		self.std = self.std.to(*args, **kwargs)
		