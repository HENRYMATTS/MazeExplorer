from ursina import * # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

app =  Ursina()

window.title = 'Maze.py'
window.borderless = False 
window.exit_button.visible = False      # Do not show the in-game red X that loses the window
window.fps_counter.enabled = False       # Show the FPS (Frames per second) counte


agent = Entity(model='cube',color=color.rgb(127,127,255),texture= 'white_cube',position=(0,1.5,0),collider = 'box',scale=1.5)

    
def input(action):
    reward = 0
    done = False
    next_state = (0,0,0)

    if action == '0' and agent.z > 0:
        agent.z -= 1*2
        rewarding()
       
        
              
    if action == '1' and agent.z < 6 :  
        agent.z += 1*2
        rewarding()
       
            
    if action == '2' and agent.x < 6:
        agent.x += 1*2
        rewarding()
        
   
    return 
   

def rewarding():
    done = False
    if agent.position == (2,1.5,2) or   agent.position == (2,1.5,6) or agent.position == (4,1.5,6)  or agent.position == (6,1.5,0):
        reward = -120
        next_state = agent.position
        done = True
    elif agent.position == (6,1.5,6):
        reward = 200
        next_state = agent.position
        done = True  
    elif agent.position == (2,1.5,4) or agent.position == (4,1.5,4)  or agent.position == (4,1.5,2)  or  agent.position == (6,1.5,2) or agent.position == (6,1.5,4):
        reward = 60
        next_state = agent.position
    else:
        reward = 0
        next_state = agent.position
    next_state = torch.tensor( next_state, dtype=torch.float)
    next_state = next_state.view(-1)
        
    
    app.step()          
    return reward,next_state ,done
      
    
          
class Blocks(Entity):
    def __init__(self, model,position):
        super().__init__()
        self.model = model
        self.scale = 2
        self.position = position
        self.texture = 'white_cube'
        self.collider = 'box'
             
for z in range(4):
    for x in range(4):
        blocks = Blocks('cube',(z*2.02,0,x*2.02))
      
        if blocks.position == (0,0,0):
            blocks.color = color.rgb(144,238,144)
        if blocks.position == (6.06,0,6.06):
            blocks.color = color.rgb(255,127,127)
        if blocks.position == (2.02,0,6.06) or blocks.position == (2.02,0,2.02) or  blocks.position == (6.06,0,0) or  blocks.position == (4.04,0,6.06):
            destroy(blocks)
            
     
EditorCamera()        

    
# Neural Network
class PolicyNet(nn.Module):
    def __init__(self,input_size,output_size):
        super(PolicyNet,self).__init__()
        self.l1 = nn.Linear(input_size,256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, output_size)

    def forward(self,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x




#hyperparameters
input_size = 3 
output_size = 3



policy = PolicyNet(input_size,output_size)
policy.load_state_dict(torch.load('models/3Dmodel1.pth'))
policy.eval()  # Set the model to evaluation mode





# Training loop

agent.position = (0,1.5,0)
state = torch.tensor( agent.position , dtype=torch.float)
state = state.view(-1)
done = False
while True:   
    with torch.no_grad():
        q_values = policy(state)
    action = torch.argmax(q_values).item()
    app.input(str(action),is_raw= True) # sends action input to game    
    reward,next_state ,done = rewarding() # type: ignore 
    state = next_state      
    if done:
        agent.position = (0,1.5,0)
        state = torch.tensor( agent.position , dtype=torch.float)
        state = state.view(-1)    
          
 
        

               

        



