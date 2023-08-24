# CausalFandom
This project is conducted to solve the following two tasks.
- Identify which on-platform user behaviours cause users to purchase and advocate more for an artist  (i.e. share their music more)
- Build a model(s) that predicts how much a listener will stream, purchase or advocate for an artist based on their on-platform behaviour.

Step 1 finish the first task, determine causes of: 1.Streaming, 2.Purchasing Merch, 3.Purchasing Tickets, 4.Sharing 

Step 2&3 finish the second task. First use causes from step 1 as features in model to predict the above 4 outcomes. 
Then use Audience Segments to predict the same outcomes. 
The comparison between the above two is conducted after that.

Step 4 jointly determine cause and effect segments using causal feature learning. 
While step 1 determined which variables are candidate causes, the aim here is to determine the range of values of those.

Extra step is to use reinforcement learning credit assignment, i.e. 
finding which behaviour caused the reward (or any part of the state) to change.


## Folder Structure
This project is plit into four steps, the code of each are as following.
- `step1/` contains all code for step 1
- `step2&3` contains all code for step 2 and step 3
- `step4` contains all code for step 4
- `extra` contains all code for extra step

## Usage Instructions
- `./step1/step1_instructions.ipynb` is the instructions to get the result of step1
- `./step2&3/step2_3_instructions.ipynb` is the instructions to get the result of step2 and step3
- `./step4/step4_instructions.ipynb` is the instructions to get the result of step4
- `./extra/step5_reinforcement_learning.ipynb` is the instructions to get the result of extra step

## File Structure
```
 repository
    |
    ├── step1
    |    ├── datatools.py
    │    └── step1_instructions.ipynb
    |
    ├── step2&3
    |    ├── func.py
    │    └── step2_3_instructions.ipynb
    |
    ├── step4
    |    ├── func.py
    │    └── step4_instructions.ipynb
    |
    └── extra
         └── step5_reinforcement_learning.ipynb
```