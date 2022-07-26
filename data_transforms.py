from torchvision import transforms

class DataTransform():

    def __init__(self):
        self.transform_dict = {
            "norm_crop": transforms.Compose([
                transforms.CenterCrop(224),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "norm_no_crop": transforms.Compose([
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        self.default = "norm_crop"

    def get(self, transform_name):
        if not(transform_name): transform_name = self.default
        return self.transform_dict[transform_name]