# Kostas Simulator for Drone

### V1.2 (25-February-2020) 
- Add crash prevention funtionality for keeping drone crashing in the boundaries. 
- The drone will still crash because of the randomness(like the wind in the real world), but not the policies.

### V1.1 (7-August-2019)
- Reconstruction
- Add distance as observation

### V1.0 (19-June-2019)
- Add different drone placed pattern
- Add Simulation Class
    - Add plot_flag, downsampling, acceleration, max_time, drone_placed_pattern in .__init
    - For drone placed pattern, we provide
        - 0 --> Random position within the cage
        - 1 --> Distributed over one edge
        - 2 --> Distributed over all edges
        - 3 --> Starting from one corner
        - 4 --> Starting from all corner
- Now the script can run in terminal, refreshing plot each step.
    - Enter command "python3 kostas.py"
    - For arguments description, Enter "python kostas.py -h"


### V0.1 (18-June-2019)
- Change the action_id in GeneralMissionParameters, the action could be assigned in the function
- Reconstruct whole function.

### V0.0 (17-June-2019)
- Add random action
- Add plot flag, down sampling for acceleration
- Bug fixed
