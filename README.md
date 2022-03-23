#Project description
This work was done under Professor Roger Grosse, Guodong Zhang, and Chaoqi Wang during the year 2021. We set out to 
explore how overparameterization of CNNs interacts with pruning. The way we overparameterize our networks whilst keeping 
the architecture the same is by changing the number of channels between convolutional layers (i.e. increasing the number 
of feature maps) which in turn increases the number of model parameters but keeping the general architecture the same. 
Additionally, we ran a lot of empirical tests to further explore this idea by pruning before/after training and using 
different (at the time) SOTA algorithms such as SynFlow and SNIP as well as some of the classical magnitude pruning 
algorithms. We found that the iterative magnitude pruning algorithm ended up working best after which we turned our 
attention to network training dynamics. We explored this idea using a teacher-student setting, however, the project had to 
come to a stop here due to personal circumstances. I've pushed a few of our plots to further understand our findings.
