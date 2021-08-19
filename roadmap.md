## Paper :: 1 :: Speed predictions from dash-cam video

- Coding NNs dont help here.
- Conventional Methods are best in this regard.
  - Explain the process which leads to acceptable answers.
- results


## Paper :: 2 ::  Pitch and Yaw predictions from dash-cam video

- Same thing we learned from NNs from the first paper
  - Conventional method to find VP
  - From VP, find the pitch and yaw using eqn.


## Paper :: 3 :: Crack predictions from UAVs on-board Camera

- These are the areas where NNs shine best.
- End-2-End approach where the search region is much smaller.
    - meaning : Driving on a 2D plane is much more easy to predict for the
      search spaces in term of Accel<->Brake and Steer.



## Thesis :: EfficientNet-B2 model for training End-2-End models


--> DataLoading :: Comma Dataset, Honda Dataset, and any other dataset I can
find

--> Model Definition :: use the code from PyTorch Template on Git

--> How to Train the model ??
    1. Only on steering without corrections -- assuming constant forward
       velocity
    --> compare the results with Nvidia Paper

    2. Steering and Accel<->Brake  
 
