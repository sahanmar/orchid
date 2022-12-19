class InvalidModelName(Exception):
    def __init__(self, model_id: str):
        super().__init__(
            f"Amigo, bad news... {model_id} is not on the table..."
        )
