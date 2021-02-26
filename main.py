import datasets
from keras_segmentation.models.unet import resnet50_unet

n_classes = 6
n_epochs = 30


dataset = 'dataset-medium' # 9.0 GB download

datasets.download_dataset(dataset)

# train the model
model = resnet50_unet(n_classes=n_classes ,  input_height=256, input_width=256)

model.train( 
train_images =  "dd-ml-segmentation-benchmark/image-chips/",
train_annotations = "dd-ml-segmentation-benchmark/label-chips/",
checkpoints_path = "resnet50_unet" , epochs=n_epochs
)
