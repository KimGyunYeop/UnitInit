# /bin/bash

python -u test_image_classification.py --result_path vit --image_classification_dataset cifar10 --no_add_linear --batch_size 64 --gpu 0 --dev 
python -u test_image_classification.py --result_path vit --image_classification_dataset cifar100 --no_add_linear --batch_size 64 --gpu 0 --dev 
python -u test_image_classification.py --result_path vit --image_classification_dataset imagenet-1k --no_add_linear --batch_size 64 --gpu 0 --dev 

python -u test_image_classification.py --result_path convnext --image_classification_dataset cifar10 --no_add_linear --batch_size 64 --gpu 0 --dev 
python -u test_image_classification.py --result_path convnext --image_classification_dataset cifar100 --no_add_linear --batch_size 64 --gpu 0 --dev 
python -u test_image_classification.py --result_path convnext --image_classification_dataset imagenet-1k --no_add_linear --batch_size 64 --gpu 0 --dev 



