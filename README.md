# -MazeExplorer
If you reading this then chances are you were influenced by my social media post good thing but before we head on any further, i have to clarify on a few things .
I couldn't explain everything involved in  the process of reinforcement learning (RL) all in a single post so  i was very  very abstract.
This means that there is a bit more to the training process than i revealed but it is nothing to fear if you have reached this far .
I have structured the code in a way that will help you to intuitively understand the different pieces of the RL algorithm and how they fit together to produce the self learning networks.

So there are two python scripts one for training the model and another for testing
    When training is done, the weights of the model are stored in the models' folder 
    There are already two models from my training so if you want to add yours change the model name in training script in this line --->   torch.save(policy.state_dict(), 'models/3Dmodel1.pth')
     To test your model after training modify the testing script by adding your model name to this line ----> policy.load_state_dict(torch.load('models/3Dmodel1.pth'))   

Hardware requirements:
      The model is very much capable of training on cpu  (even dual core pcs) so there is no need for a gpu because i  built this project for Education purporses only to guide those who would       wish to get started with RL. For more real world advanced projects GPUs are used for training  because they are faster

In conclusion , i hope i've simplified the process if not better next time       
