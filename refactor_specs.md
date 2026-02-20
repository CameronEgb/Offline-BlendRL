### Experiment entry point

Experimental workflow will consist of changing parameters at the top of the entry point script to control which methods get run on which data. 

entry point command:
    ./run_full_cycle.sh

run_full_cycle.sh will send off batch jobs for either, or both, online and offline RL training for the given input variables. 

experimentid : string
environment : string
online methods : string array
online steps : string array
use large dataset path: boolean
offline methods : string array
offline datasets : string array
offline epochs : int
intervals count : int

the script needs to be robust and handle the following cases in the following ways:


for specified online methods (should be one of: ppo, blendRL-ppo, blendRL-iql, iql), the script should run an online training job for each method and gather and save the necessary data for plotting (see plotting). 

when offline methods are specified, batch jobs should be run for offline trianing, in particular the offline method should trian on the datasets specified jointly by the experimentid and the offline datasets. The offline datasets hyperparam will be a string array which specifies by which method the dataset should have been generated and in which experimentid it was generated. 

For offline trianing, if the dataset does not exist, that should be output and the job skipped, however if the data will exist (i.e. it is specified to run and generate in the script), then send the offline training off as a dependent job. 

if a specified experimentID already exists, for each individual method specified that also already has been run before, the script should prompt the user y/n to rerun and overwrite, or to skip, that specific job. to clarify, if the combination of experimentID, methodname, (and datasets in the offline case), hasnt been run yet, send off the job, otherwise ask what should be done. 

### Data organization and saving

There should be a single top level dir holding all data made by each experiments s.t. each experiment is a subdir with the experimentID. This subdir should hold all results necessary for plotting (see plotting), and should be where data gets saved to. The only thing to fix here is that sometimes for larger datasets, if the use large dataset path boolean is set to true, then for the specified large dataset path (ask for it at the very end), the data should be handled to save gracefully, and be accessed, in that location.

### Plotting

After all experiments are said and done, I will want to plot the following data. (I will also want to be able to plot data while it is still generating too with whatever is saved so far). please make scripts for the following:
- Online training returns (episodic and by step) (standard plot)
- Offline training returns by dataset size:
    I will want a plot of training efficiency for the offline method. Therefore, according to the intervals count hyperparam, for online methods stop training that many times (spaced evenly) to run the model made so far (100 episode average reward) to generate points to compare to the offline method. For the offline trianing, for each interval, train only on the data the online method has found so far in each interval. (So train on growing datasets) Clarify this graph if you need to, it is the trickiest.
- Trianing time tables. Record training times for each interval, and overall end to end, so a table can be output with all the wallclock times for each run.


### Auxillary features to make

Sync-logs: scp all files that dont already exist from data logging from remote to local (see old sync logs script). 
Kill-experiments: Make sure to save experiment process, or job ids when you start them off so that if i run ./kill_experiments "experimentID" the jobs, (or processes in the local case) should get canceled, and delete the logs for it. 
Daisy Chaining: To circumvent the 6 hr runtime, please save all necessary parts of training states and resume trianings in a new batch jobs, 10 minutes before the limit on the jobs finish.  
Flag for local testing: If i give the --local flag to the ./run_full_cycle.sh script, then start training locally and not with sbatch. 