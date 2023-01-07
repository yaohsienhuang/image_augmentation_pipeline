import imgaug.augmenters as iaa

class IaaSequence:
    def __init__(self,probility):
        '''
        ref : https://github.com/aleju/imgaug
        '''
        sometimes = lambda aug: iaa.Sometimes(probility, aug)
        self.sequence = iaa.Sequential([
                #sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255), per_channel=0.5)),
                #sometimes(iaa.ContrastNormalization((0.5, 2.0), per_channel=1),),
                sometimes(iaa.Fliplr(0.6)),
                sometimes(iaa.Flipud(0.6)),
                sometimes(iaa.ChannelShuffle(0.2)),
                sometimes(iaa.GaussianBlur(sigma=0.8)),
                #sometimes(iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.2, 0.3), per_channel=True))
                #sometimes(iaa.Emboss(alpha=(0, 0.3), strength=(0, 0.5))), # emboss images
                ],random_order=True)
        
class IaaAugmentation(IaaSequence):
    def __init__(self,probility):
        super().__init__(probility)

    def __call__(self, image):
        image = self.sequence.augment_image(image)
        return image