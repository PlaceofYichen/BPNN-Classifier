# Neural Network Classifier to Identify Karst Skinholes from LiDAR Data

## An extra credit project assignment for Temple University CIS-4525 Machine Learning course.

### Project descriptions
[embed]https://github.com/PlaceofYichen/cis4526-Extra/blob/master/PA%204.pdf[/embed]

Dependency: matplotlib
1. Adjust the hyperparameter of BP network : 
    10-hiden 
    net = BPNNet(num, 10, 1); 
    net.train(train_data,iterations=1000, N=0.01, M=0.1);
2. Simply run 'pa4.py' and wait (for about 30 minutes);
3. For the results, negative values indicate that there is no skinhole; 
   for positive values, the closer it is to 1, the larger chance there exists a skinhole.

![Training Performance](https://github.com/PlaceofYichen/cis4526-Extra/blob/master/TrainingPerformance.png)



