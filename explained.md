# Soft Actor-Critic Methods

Soft Actor-Critic methods are a type of reinforcement learning algorithms that hold a special spot in my heart. I remember the excited, and slightly sinking, feeling of "wow, physics really is ubiquitous" when I heard that entropy was used as guiding metric for agents to learn how to play games.

SAC methods was also the first big reinforcement learning project I undertook from just reading the paper and without having an identical project to reference against use to check my answers. It was therefore the first one I had to spend hours twiddling hyper-parameters and checking for bugs. I eventually ended up with quite well commented code, a working test set and a pretty good understanding of soft actor-critic. This is therefore what I am going to go through today.

## Overview

### What's an actor-critic?

You can think of actor-critic methods as like a newbie player training with their new coach. The player attempts to make a play and the coach tells them whether it was any good or not. If it was, the player will be more likely to make that play again. If not, less likely.

In the case of actor critic, the player is a neural network which is learning to map states to good actions and the critic is another neural network learning to score how good these are. 

### What's so soft about soft actor-critic?

Soft actor-critic is 'soft' as it looks to maxmimise entropy as well as reward in its policy's objective function. This is 'soft' as the actor may be encouraged to choose a slightly smaller expected reward path, as long as it is more entropic.

### Why entropy?

Shannon's entropy is the measure of uncertainty in a probability distribution.