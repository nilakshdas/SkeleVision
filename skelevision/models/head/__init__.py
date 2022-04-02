from .keypoint import ModularConvDeconvInterpKeypointHead

KEYPOINT = {"ModularConvDeconvInterpKeypointHead": ModularConvDeconvInterpKeypointHead}


def get_keypoint_head(name, **kwargs):
    return KEYPOINT[name](**kwargs)
