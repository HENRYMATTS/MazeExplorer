from ursina import * # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


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


# Experience Replay
class Experience():
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


#hyperparameters
input_size = 3 
output_size = 3
gamma = 0.9
epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.99965
learning_rate = 0.001
replay_capacity = 100000
batch_size = 1000


policy = PolicyNet(input_size,output_size)
target = PolicyNet(input_size,output_size)
target.load_state_dict(policy.state_dict())  # Copy policy weights to target network
target.eval()  # Set target network to evaluation mode

replay = Experience(replay_capacity)

# loss $ optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Training loop
for episodes in range(1800):
    agent.position = (0,1.5,0)
    state = torch.tensor( agent.position , dtype=torch.float)
    state = state.view(-1)
    done = False
    counter = 0
    for episode in range(160):
        
        if random.random() < epsilon:
           
            action = random.randint(0,2)
        else:
            with torch.no_grad():
                q_values = policy(state)
                action = torch.argmax(q_values).item()
        app.input(str(action),is_raw= True) # sends action input to game
        
        reward,next_state ,done = rewarding() # type: ignore
        replay.push(state,torch.tensor([action]),torch.tensor([reward]), next_state) 
        
        batch = replay.sample(1)
        
        if len(replay.memory) >= batch_size:
            batch = replay.sample(24)
          
            
        states,actions,rewards,next_states = zip(*batch) # unpacking
           
           
        states = torch.stack(states).squeeze(1)
        actions = torch.cat(actions).unsqueeze(1)
        rewards = torch.cat(rewards)
        next_states = torch.stack(next_states).squeeze(1)
        
        q_values = policy(states).gather(1, actions)
        q_values =  q_values.squeeze(1)
       
           
           
        max_q_values = target(next_states).max(1)[0]
        target_q_values = (rewards + gamma * max_q_values)
       

        
      
        loss = criterion(q_values,  target_q_values)
        optimizer.zero_grad()    
        loss.backward()
        optimizer.step()
               
        state = next_state
        epsilon = max(epsilon * epsilon_decay, epsilon_min) 
        counter += 1 
        if counter == 15:
            target.load_state_dict(policy.state_dict())
            
            counter = 0
        
          
        state = next_state
        if done:
            break
               
torch.save(policy.state_dict(), 'models/3Dmodel1.pth')  
        



