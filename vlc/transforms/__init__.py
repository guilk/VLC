from .mae_transforms import (
    mae_transform,
    mae_transform_randaug,
    mae_transform_test
)

_transforms = {
    "mae": mae_transform,
    "mae_randaug": mae_transform_randaug,
    "mae_test": mae_transform_test
}


def keys_to_transforms(keys: list, size=224):
    return [_transforms[key](size=size) for key in keys]
