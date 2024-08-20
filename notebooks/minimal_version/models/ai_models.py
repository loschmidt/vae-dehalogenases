from typing import Tuple

from notebooks.minimal_version.models.variational_autoencoder import VariationalAutoencoder


class AIModel:
    def __init__(self, model_config):
        self.model_config = model_config
        self.model = model_config['model']

    def get_model(self) -> Tuple[VariationalAutoencoder, bool]:
        """
        return model and flag if it requires labels or not
        """
        if self.model in ['vae', 'vae_conditional']:
            return VariationalAutoencoder(self.model_config), False if self.model == 'vae' else True
