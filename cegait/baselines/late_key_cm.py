from ..models.key_conditioned_protector import KeyConditionedProtector


class LateKeyCM(KeyConditionedProtector):
    def __init__(self, **kwargs):
        super().__init__(early_key=False, **kwargs)
