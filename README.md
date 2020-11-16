# BP Neural Network Classifier to Identify Karst Skinholes from LiDAR Data

##### *An extra credit assignment for Temple University CIS4526 Machine Learning course.*

## Project descriptions
Please see [descriptions](https://github.com/PlaceofYichen/BPNN-Classifier/tree/master/descriptions) for details.
![descriptions](/descriptions/pa4.png)

*Dependency: matplotlib*
1. Adjust the hyperparameter of BP network : 
    - 10-hiden 
    - net = BPNNet(num, 10, 1); 
    - net.train(train_data, iterations=1000, N=0.01, M=0.1);
2. Simply run 'pa4.py';
3. Results are saved in an auto-generated file called "pred.csv". For values:
    - positive : the closer it is to 1, the larger chance there exists a skinhole;
    - negative: there is no skinhole.

## Training Performance
![Training Performance](https://github.com/PlaceofYichen/cis4526-Extra/blob/master/TrainingPerformance.png)
