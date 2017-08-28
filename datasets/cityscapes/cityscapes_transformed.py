from datasets.cityscapes.cityscapes_semantic_segmentation_dataset import CityscapesSemanticSegmentationDataset
from chainer.dataset import TransformDataset


def transform(inputs):
    img, label = inputs


class CityscapesTransformed(TransformDataset):

    def __init__(self, data_dir, label_resolution, split, ignore_labels):

        self.dataset = CityscapesSemanticSegmentationDataset(data_dir, label_resolution, split, ignore_labels)
