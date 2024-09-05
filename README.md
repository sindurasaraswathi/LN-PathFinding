# An Exposition of Pathfinding Strategies Within Lightning Network Clients

This repository provides a simulation framework for evaluating the performance of various Lightning Network (LN) clients (LND, CLN, LDK, and Eclair) in routing payments. It includes tools for running simulations, configuring parameters, and analyzing the results.

## Contents
* run_simulation.py: Main script to run payment simulations.
* config.ini: Configuration file for simulation parameters.
* analysis.py: Script for analyzing and visualizing simulation results.
* analysis_config.ini: Configuration file for analyzing the results.
* LN_snapshot.csv: Lightning Network snapshot used in the simulations.

## Usage
### run_simulation.py
This script simulates payment attempts on the LN using various client implementations.

#### Functionality:

* Randomly selects source and target nodes and payment amounts for a specified number of iterations.
* Source and destination nodes are chosen from one of the following categories:
  - Well-connected: Nodes with a total channel capacity ≥ $10^6$ and more than 5 channels.
  - Fairly connected: Nodes with a total channel capacity between $10^4$ and $10^6$, and more than 5 channels.
  - Poorly connected: Nodes with fewer than 5 channels.
* Payment amounts are either selected randomly or fixed. For random amounts, values are chosen from bins ($10^k$ to $10^{(k+1)}$) where k ranges from 0 to the specified end range.
* The simulation runs for the selected clients and specified parameters, using the client's pathfinding strategy to determine the route for each payment.
* Payment success or failure, along with metrics such as path, total fee, delay, path length, and success/failure status, is recorded in a <filename>.csv file.

### config.ini
This configuration file allows to customize the behavior of run_simulation.py by setting parameters for the simulations.

#### Configuration Options:
General:
- cbr: Current block height.
- iterations: Number of iterations to run.
- filename: Filename for saving the results (e.g., results.csv).
- source_type: Choose node selection strategy (random, well, fair, or poor).
- target_type: Choose destination node selection strategy (random, well, fair, or poor).
- amount_type: Choose payment amount selection (random or fixed).
- amount: Fixed payment amount to route (used if amount_type is fixed).
- amt_end_range: Maximum power range for payment amounts (e.g., 8 for maximum payment of 10^8, used if amount_type is random).
- datasampling: Choose distribution type (bimodal or uniform) for channel balance sampling.
- algos: Specify which LN clients to test, separated by "|" (e.g., LND|CLN|LDK|Eclair or just CLN).
- lndcase: Specify the LND cases to test, separated by "|". Use LND1 for 'Apriori' and LND2 for 'Bimodal' probability estimations (e.g., LND1|LND2)
- eclaircase: Specify the Eclair cases to test, separated by "|". Use Eclair_case1, Eclair_case2, and Eclair_case3 for cases that employ Weight Ratios, Heuristics without logarithm, and
  Heuristics with logarithm, respectively.s (e.g., Eclair_case1|Eclair_case2|Eclair_case3)

### analysis.py
This script is used to visualize and analyze the simulation results.

#### Functionality:

Reads the simulation results from <filename.csv> and generates insights based on routing success, fee ratios, path lengths, and timelocks.

### analysis_config.ini
Configure the analysis process by adjusting the following parameters:

- filepath: Path to the simulation result file (<filename.csv>).
- amt_start_range: The starting value (k) for the payment amount range ($10^k$).
- amt_end_range: The ending value (k) for the payment amount range ($10^k$).
- amt_range_step: Step size for varying bins (e.g., $10^{(i+step)}$ where i varies from amt_start_range to amt_end_range).
- no_of_clients_success: Specify the minimum number of clients required to succeed in a transaction to include it in fee ratio analysis.

## How to Run the Simulation
* Configure your simulation parameters in config.ini.
* Run the simulation using:
```
python run_simulation.py
```
* After the simulation completes, analyze the results with:
```
python analysis.py
```
        
