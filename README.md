# An Exposition of PathfindingAn Exposition of Pathfinding Strategies Within Lightning Network Clients

To run the simulations, first configure the file config.ini. This configuration file includes default values for design elements, organized by LN client name. Additionally, there is a 'General' section for general settings.

In this section: cbr takes the current block height. source and target can be set to 'random', 'well', 'fair', or 'poor'. If 'random', both source and target will be chosen randomly. If 'well', 'fair', or 'poor', the source and target will be selected from well-connected, fairly connected, or poorly connected node categories, respectively. amount_type can be 'random' or 'fixed'. If 'random', payment amounts will be chosen randomly from 0 to $10^(amt_end_range)$. In this case, the code selects amounts from different bins depending on the iteration.

Run run_simulation.py

Edit config file as per the requirement

Results are written to a csv file

In the csv file, the values in the columns 'LND', 'LDK', 'CLN', 'Eclair_case1', 'Eclair_case2', and 'Eclair_case3' represent [path, total_fee, total_delay, path_length, Success/Failure] respectively

In the configuration file, 

        source and target types can be well, fair, poor or random.

        amount type can be fixed or random
        
