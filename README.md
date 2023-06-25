# Auto-focusing
Try to find the location of the maximum in a noisy Gaussian function


Method1:  Traditional Gradient Descent---complete---see gradient search.py


Method2:  PSO---complete---see PSO.py


Method3:  Genetic PSO---complete---see PSO_optimised.py


Method4:  Gradient-like PSO---complete---see PSO_gradient.py
We implemented an optimized Particle Swarm Intelligence using evolutionary strategies. , which is called EPSO. Particle Swarm Optimization (PSO) is done by having a population of candidate solutions, known as particles, and moving these particles around in the search space according to simple mathematical formulae over the particle's position and velocity. The Evolutionary Particle Swarm Optimization (EPSO) algorithm incorporates evolutionary algorithm principles into traditional PSO, introducing components like mutation and crossover operators from genetic algorithms to increase diversity and avoid premature convergence, thus potentially providing a better balance between exploration (global search) and exploitation (local search) compared to standard PSO. We also replace the initial velocity part when updating velocity with the gradient part, which is the gradient of fitness values between the current and previous iterations. We optimized parameters by grid search and greedy thinking. We fix all the other parameters and use the best performance value with grid search. After optimizing all the parameters, we do it again to increase the accuracy.

Method5:  Bayesian Optimisation(code from bayesian_opt library)---complete---see folder bayes_opt, run gaussian.py


Method6:  Random Forests---incorrect version & complete---a try on supervised learning---see Ensemble method.py


Method7:  CNN---working on---see CNN.py
We implemented a CNN method to predict the maximum location. We use the grid search to fetch the intensities and further fit the function. We try different kinds of structures to handle the regression task, including LeNet-5-like structure[3], AlexNet-like structure[4], VGGNet-like structure[5], and so on. There may be significant differences between different CNN structures. We found that an AlexNet-like design will have fewer measurements and found the target rapidly in high SNR. We also tried to compare the performance between Convo1d and Convo3d structures, where the neural network has or doesn't have access to space information. We found that Convo3d performs better than Convo1d with less training data. To further increase the accuracy\speed of CNN, hopeful methods include transfer learning (knowledge distillation) and ensemble methods (different structures random voting). Another way is to make the process of collecting data more suitable for fitting instead of searching.


[3] LeCun Y. LeNet-5, convolutional neural networks[J]. URL: http://yann. lecun. com/exdb/lenet, 2015, 20(5): 14.
[4] Singh I, Goyal G, Chandel A. AlexNet architecture based convolutional neural network for toxic comments classification[J]. Journal of King Saud University-Computer and Information Sciences, 2022, 34(9): 7547-7558.
[5] Wang L, Guo S, Huang W, et al. Places205-vggnet models for scene recognition[J]. arXiv preprint arXiv:1508.01667, 2015.


Method8: Reinforcement learning

![All in whole SNR](https://user-images.githubusercontent.com/111651791/226904014-3fa14d32-c8ce-43f6-b2bc-3dfa703cca0b.jpg)
