# Assignment 1: Shallow Neural Networks    

### Introduction   
In this assignment, we were tasked with building a shallow neural network on a wine dataset. After creating a basic model, we were to 
1. Tune the weights and biases and plot parameter values vs accuracy 
2. Plot percentage of training set vs accuracy
3. Plot percentage of randomly selected features vs accuracy    


### File Structure    
```bash
.
├── HTML
│   ├── Part1_ShallowNeuralNetwork.html
│   ├── Part2_PercentTraining.html
│   └── Part3_FeatureSelection.html
├── README.md
├── src
│   ├── Part1_ShallowNeuralNetwork.ipynb
│   ├── Part2_PercentTraining.ipynb
│   └── Part3_FeatureSelection.ipynb
└── weights
    ├── model_part1.hdf5
    ├── model_part1_a.hdf5
    ├── model_part2.hdf5
    └── model_part3.hdf5

```
### Description of File Structure 
1. ***HTML*** - HTML Files for easier view of notebooks 
1. ***src/Part1_ShallowNeuralNetwork*** - Verbose notebook that includes basic neural network, tuning weights, and plotting parameter values 
2. ***src/Part2_PercentTraining***- includes plot of percentage training vs accuracy 
3. ***src/Part3_FeatureSelection*** - includes plot of randomly selected features vs accuracy   
4. ***weights/model_part1.hdf5*** - saved weights/bias for part 1 
5. ***weights/model_part1_a.hdf5*** - saved weights/bias for part 1 for learning rate
6. ***weights/model_part1_b.hdf5*** - saved weights/bias for part 1 for batch size 
5. ***weights/model_part2.hdf5*** - saved weights/bias for part 2 
6. ***weights/model_part3.hdf5*** - saved weights/bias for part 3 

### Methods and Results 
In my solution, I build a simple neural network that utilizes the callback function to save the most optimal weights and biases of the trained model to a .hdf5 file. During testing, I call this file to build the testing model and evaluate the accuracy. I also plot learning rate vs accuracy and batch size vs accuracy within the first notebook. 

In part 2, our task is to plot different percentages of training sizes vs the accuracy of our model. The motivation is to try to find what the most optimal training size is for our model. Several research papers illustrate that an ideal percentage is around 70%-80% training size and 30%-20% validation size. The notebook reflects these findings.    

In part 3, our task is to plot randomly selected features vs the accuracy of our model. I utilized Recursive Feature Elimanation using Random Forest Classifier in order to get the percentage of features needed. RFE works by eliminating teh least important features. It continues receursively until the specified number of features is reached.The motivation for feature selection is that it enables our model to train faster as it reduces the complexity of the model. My work suggests that the most optimal percentage of features to use is 70-80%. 

   
