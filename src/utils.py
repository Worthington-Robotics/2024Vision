from typing import List

class MovingAverage:
	n: int
	items: List[float] = []

	def __init__(self, n: int):
		self.n = n

	def add(self, item: float):
		self.items.append(item)
		if len(self.items) > self.n:
			self.items.pop(0)

	def average(self) -> float:
		return sum(self.items) / float(self.n)
