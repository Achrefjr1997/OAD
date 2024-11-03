from torchgeo.models import vit_small_patch16_224,ViTSmall16_Weights
import timm

def OADModel(num_classes):
    model = vit_small_patch16_224(weights=ViTSmall16_Weights.SENTINEL2_ALL_DINO,num_classes=num_classes)
    return model
