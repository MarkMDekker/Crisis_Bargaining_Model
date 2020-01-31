# Crisis_Bargaining_Model
DOI: will follow

Some general text about the model

# Versions
31/01/2020 - Initial commits and first draft of readme. Let me know if the code works on your computers.

# Explanation (write with final version code)
As input, use the config.ini file. The following parameters can be set there:
- AUDIENCECOST_A_S: the audience payoff a_s, the benefit (for sender) due to a successful treat.
- AUDIENCECOST_A_T: the audience cost a_t, the cost (for target) due to conceding to the threat.
- AUDIENCECOST_B_S: the audience cost b_s, the cost (for sender) of backing down (not engaging in conflict)
- AUDIENCECOST_B_T: the audience payoff b_t, the benefit (for target) due to the sender's backing down
- ECONONOMICCOSTS_S: the economic costs c_s for the sender of engaging in conflict
- ECONONOMICCOSTS_T: the economic costs c_t for the target of engaging in conflict
- GLOBAL_STD: one uncertainty level for all payoffs of the opponents and one own's economic costs of engaging in conflict.

For a single run, one only needs to fill in the config.ini as desired, and run the '''__Main__.py''' file by typing into a terminal: '''python __Main__.py'''

# Installation requirements
Python version 3.7

Package requirements:
- numpy
- configparser
- scipy

# Assumptions
We have assumed here that unknown variables are derived from a Gaussian distribution with mean equal to the true value of the variable, and some specified standard deviation.

# Acknowledgements
Authors, etc.
