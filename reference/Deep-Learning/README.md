# CS583 Deep Learning Assignments 


#### Assignment 1: Shallow Neural Networks 
In this assignment, we were tasked with building a shallow neural network on a wine dataset. After creating a basic model, we were to 
1. Tune the weights and biases and plot parameter values vs accuracy 
2. Plot percentage of training set vs accuracy
3. Plot percentage of randomly selected features vs accuracy   

####  Assignment 2: Evaluation and Error Analysis 
In this assignment, we are tasked at applying a CNN for regression on a 4D array of rotated images. Our goal is to predict angles of rotated handwritten digits. 

Since the assignment requires a specific matlab dataset, I wrote the assignment in matab. 

#### Assignment 3: Machine Translation 
In this assignment, we are tasked to find and collect as many as possible foreign input sentences that Google Translate will make an error on. 

Ideas of improving the MT could be adding Residual LSTM and maybe including more colloquial corpus from online forums. 

#### Repo Structure 
```bash 
.
├── HW1
│   ├── HTML
│   │   ├── Part1_ShallowNeuralNetwork.html
│   │   ├── Part2_PercentTraining.html
│   │   └── Part3_FeatureSelection.html
│   ├── README.md
│   ├── cquinto_hw1.zip
│   ├── src
│   │   ├── Part1_ShallowNeuralNetwork.ipynb
│   │   ├── Part2_PercentTraining.ipynb
│   │   ├── Part3_FeatureSelection.ipynb
│   │   ├── model_part1_a.hdf5
│   │   ├── model_part2.hdf5
│   │   └── model_part3.hdf5
│   └── weights
│       ├── model_part1.hdf5
│       ├── model_part1_a.hdf5
│       ├── model_part1_b.hdf5
│       ├── model_part2.hdf5
│       └── model_part3.hdf5
├── HW2
│   ├── README.md
│   ├── cs583_cquinto_hw2.zip
│   ├── diaryFile
│   ├── main.m
│   ├── main.m~
│   └── reports
│       ├── normalization.png
│       ├── training.png
│       └── worst5predictions.png
├── HW3
│   └── HW3_AdversarialMT.pdf
└── README.md
```
