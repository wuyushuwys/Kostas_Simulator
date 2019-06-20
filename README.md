# Kostas Simulator for Drone

## Version 4 (19-June-2019)
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


## Version 3 (18-June-2019)
- Change the action_id in GeneralMissionParameters, the action could be assigned in the function
- Reconstruct whole function.

## Version 2 (17-June-2019)
- Add random action
- Add plot flag, down sampling for acceleration
- Bug fixed
