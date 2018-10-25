
### __Implementation of defect detection with transfer learning using PyTorch__

To achieve defect detection, you could first use the pre-trained Inception V3 model without freezing any layers. Only the last fully connected layer is going to be modified to fit the two-class defect classification problem. Then, you could feed your own dataset to retrain the entire net. After the process of retraining, the net can learn features from fed images.  

In test phase, you will get predictions for different images.

An example of training and testing usage is shown as follows:  
* __training__:  
python train_v.py --restore=0 --batch_size=64 --epochs=50

* __testing__:  
python test_v.py
