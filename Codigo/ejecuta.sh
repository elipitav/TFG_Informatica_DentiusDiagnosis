#!/bin/bash
    #SBATCH --job-name=resnet34ThreeChannel                                 # Job name
    #SBATCH --nodes=1                                                       # -N Run all processes on a single node   
    #SBATCH --ntasks=1                                                      # -n Run a single task   
    #SBATCH --cpus-per-task=2                                               # -c Run 2 processors per task       
    #SBATCH --mem=8gb                                                       # Job memory request
    #SBATCH --time=00:30:00                                                 # Time limit hrs:min:sec
    #SBATCH --output=prueba_resnet34.log                                    # Standard output and error log
    
    source ../miniconda3/bin/activate
    python lanzador.py -p -f conf.json
