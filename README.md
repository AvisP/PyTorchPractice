# PyTorchPractice
Practicing pytorch by understanding others code

<h3>Variational Encoder on MNIST</h3>
  
 Training and visulization of results of a Variational Auto Encoder based on the example in https://github.com/pytorch/examples/tree/master/vae
 

Iteration 1            |  Iteration 24    | Iteration 50 
:-------------------------:|:-------------------------:|:-------------------------:
![VAE_Iter1](https://github.com/AvisP/PyTorchPractice/blob/master/VAE_MNIST/results/sample_1.png) | ![VAE_Iter24](https://github.com/AvisP/PyTorchPractice/blob/master/VAE_MNIST/results/sample_24.png) |  ![VAE_Iter50](https://github.com/AvisP/PyTorchPractice/blob/master/VAE_MNIST/results/sample_50.png)


<h3>CNN Layer weight and output visualization</h3>
  
  Plotted filter weights and outputs for a test image of popular pretrained convolutional neural network models ( ResNet, AlexNet) based on example in https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/  and https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e
  
 <h4>Filter Visualization of Alex Net</h4>
  
  Layer 0  (Multi Channel)    |   Layer 3  (Single Channel)      |  Layer 6   (single Channel)
:-------------------------:|:-------------------------:|:-------------------------:
![AlexNet_Layer0](https://github.com/AvisP/PyTorchPractice/blob/master/LayerVisualization/Layer0_AlexNet.png) | ![AlexNet_Layer3](https://github.com/AvisP/PyTorchPractice/blob/master/LayerVisualization/Layer3_AlexNet_SingleChannel.png) |  ![AlexNet_Layer6](https://github.com/AvisP/PyTorchPractice/blob/master/LayerVisualization/Layer6_AlexNet_SingleChannel.png)

 <h4>First layer weight of ResNet</h4>
  
<img align='center' src="https://github.com/AvisP/PyTorchPractice/blob/master/LayerVisualization/output/Resnet_weights_firstlayer.png" width="40%">

 <h4>Test Image for ResNet output</h4>
  
  <img align='center' src="https://github.com/AvisP/PyTorchPractice/blob/master/LayerVisualization/data/cat_test.jpg" width="40%" title="Test Image">
  
  
 Layer 0  (Multi Channel)    |   Layer 3  (Single Channel)      |  Layer 6   (single Channel)
:-------------------------:|:-------------------------:|:-------------------------:
<img align='center' src="https://github.com/AvisP/PyTorchPractice/blob/master/LayerVisualization/output/ResNet_layer_0.png" width="350" height="350" title="ResNet_Layer0_Output"> | <img align='center' src="https://github.com/AvisP/PyTorchPractice/blob/master/LayerVisualization/output/ResNet_layer_24.png" width="350" height="350" title="ResNet_Layer24_Output"> |  <img align='center' src="https://github.com/AvisP/PyTorchPractice/blob/master/LayerVisualization/output/ResNet_layer_48.png" width="350" height="350" title="ResNet_Layer48_Output">


  
