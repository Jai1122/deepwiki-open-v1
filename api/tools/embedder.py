import adalflow as adal

from api.config import configs


import logging
import adal

def get_embedder() -> adal.Embedder:
    embedder_config = configs["embedder"]

    # --- Initialize Embedder ---
    model_client_class = embedder_config["model_client"]
    if "initialize_kwargs" in embedder_config:
        model_client = model_client_class(**embedder_config["initialize_kwargs"])
    else:
        model_client = model_client_class()
    embedder = adal.Embedder(
        model_client=model_client,
        model_kwargs=embedder_config["model_kwargs"],
    )
    logging.info(f"Embedder methods: {dir(adal.Embedder)}")
    return embedder
