import logging
import os
import typing as tp

import spacy
import torch
from bm import events
from bm.cache import Cache, MemoryCache
from bm.utils import Frequency

from . import base

logger = logging.getLogger(__name__)

class CLIPTextEmbedding(base.Feature)

	"""
	CLIP-text embeddings for training brain module

	"""

	event_kind = "text"
	dimension = 768
	# model_size = "md" # TODO: not sure about this so commented out

	_LANG = "auto" # TODO: apparently important for memory/speed but might be audio-specific

	def __init__(self, sample_rate: Frequency, lang: str = "auto") -> None:

		super().__init__(sample_rate=sample_rate)
		self._model_cache = MemoryCache(self.__class__.__name__)

	@property
	def model_name(self):
		# TODO: placeholder until we figure out what Lang is
		return f"{VALID_SPACY_LANG[self._LANG]}_{self.model_size}"

	@property
	def cache(self):
		# Lazy attr. because model name can change on the fly
		return Cache(self.__class__.__name__, self.model_name)

	@property
	def model(self) -> tp.Any:
		try:
			return self._model_cache.get(spacy.load,name=self.model_name)
		except OSError as e:
			raise OSError(
				f'You need to run "python -m spacy download {self.model_name}"') from e

	# TODO: compute function replaced by "retrieve" since these embeddings are pregenerated for time
	# otherwise it looks like embeddings are computed using forward pass to model that generates embeddings
	# of that type
	# def _compute(self, text: str) -> torch.Tensor:

	def _retrieve(self, text: str) -> torch.Tensor:
		if not text:
			out = self.default_value # TODO: not sure what this is

		else:
			CLIP_text_latent = np.load('/mnt/sphere/projects/image_decoding_from_brain/data/extracted_features/things_cliptext.npy')
			# index by text
			# return embedding
		return out


	def get(self, event: events.Text) -> torch.Tensor:
		return self.cache.get(self._compute,text=event.text)
