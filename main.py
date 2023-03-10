import argparse
from collections import namedtuple
import configparser

from neat_dynamics.env.lundar_lander import LundarLanderNovelty
from neat_dynamics.env.cartpole import CartPole

def parse_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    config_dict = {}
    
    config_dict["lr"] = float(config["DEFAULT"]["lr"])
    config_dict["ensemble_size"] = int(config["DEFAULT"]["ensemble_size"])
    config_dict["memory_capacity"] = int(config["DEFAULT"]["memory_capacity"])
    config_dict["hidden_size"] = int(config["DEFAULT"]["hidden_size"])
    config_dict["num_hidden"] = int(config["DEFAULT"]["num_hidden"])
    config_dict["novel_threshold"] = float(config["DEFAULT"]["novel_threshold"])
    config_dict["novelty_neighbors"] = int(config["DEFAULT"]["novelty_neighbors"])
    config_dict["min_archive_size"] = int(config["DEFAULT"]["min_archive_size"])
    config_dict["min_reproduce"] = int(config["DEFAULT"]["min_reproduce"])



    config = namedtuple("GenericDict", config_dict.keys())(**dict(config_dict.items()))

    return config


def main(args):
    config = parse_config("config.ini")
    if args.env == "cartpole":
        env = CartPole(args, config)
    else: 
        env = LundarLanderNovelty(args, config)

    for i in range(100000):
        print("\nGENERATION", i)
        env.eval_population()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--init_weight_mean", type=float, default=0.0, 
        help="Mean of initial weight")
    parser.add_argument("--init_weight_std", type=float, default=0.4, 
        help="Std of initial weight")
    parser.add_argument("--weight_max", type=float, default=100.0, 
        help="Maximum value of weight.")
    parser.add_argument("--weight_min", type=float, default=-100.0, 
        help="Minimum value of weight.")
    
    parser.add_argument("--init_bias_mean", type=float, default=0.0, 
        help="Mean of initial bias")
    parser.add_argument("--init_bias_std", type=float, default=0.4, 
        help="Std of initial bias")
    parser.add_argument("--bias_max", type=float, default=100.0, 
        help="Maximum value of bias.")
    parser.add_argument("--bias_min", type=float, default=-100.0, 
        help="Minimum value of bias.")

    parser.add_argument("--mutate_link_weight_rate", type=float, default=0.8, 
        help="Probability of mutating all link weights.")
    parser.add_argument("--mutate_link_weight_rand_rate", type=float, default=0.05, 
        help="Likelihood of randomly initializing new link weight.")
    parser.add_argument("--mutate_weight_power", type=float, default=0.4, 
        help="Power of mutating a weight.")
    parser.add_argument("--mutate_add_node_rate", type=float, default=0.15, 
        help="Likelihood of randomly adding a new node.")
    parser.add_argument("--mutate_add_link_rate", type=float, default=0.30, 
        help="Likelihood of randomly adding a new link.")
    parser.add_argument("--mutate_enable_gene", type=float, default=0.25, 
        help="Likelihood of randomly enabling a gene.")
    parser.add_argument("--mutate_no_crossover", type=float, default=0.1, 
        help="Likelihood of copying a parent without crossover.")
    parser.add_argument("--mutate_add_recur_rate", type=float, default=0.05, 
        help="Likelihood of adding a recurrent link.")
    parser.add_argument("--reproduce_avg_trait_rate", type=float, default=0.5, 
        help="Likelihood of averaging the parents traits.")
    parser.add_argument("--reproduce_interspecies_rate", type=float, default=0.001, 
        help="Likelihood of reproducing across species.")


    parser.add_argument("--speciate_disjoint_factor", type=float, default=1.0, 
        help="Gene disjoint factor used for comparing two genotypes.")
    parser.add_argument("--speciate_weight_factor", type=float, default=3.0, 
        help="Gene trait weight factor used for comparing two genotypes.")
    parser.add_argument("--speciate_compat_threshold", type=float, default=3.0, 
        help="Gene trait weight factor used for comparing two genotypes.")
    parser.add_argument("--respeciate_size", type=int, default=2, 
        help="Size for respeciation.")
    parser.add_argument("--max_species", type=int, default=1, 
        help="Size for respeciation.")
    

    parser.add_argument("--init_pop_size", type=int, default=150, 
        help="Initial population size.")
    parser.add_argument("--survival_rate", type=float, default=0.2, 
        help="Percentage of organisms that will survive.")
    parser.add_argument("--env", default="cartpole", 
        help="Environment to run..")
    parser.add_argument("--max_stagnation", type=int, default=20,
        help="Maximum number of stagnation generations before the species is terminated.")
    parser.add_argument("--elites", type=int, default=2,
        help="Number of elites to preserve if a species is terminated.")

    parser.add_argument("--novelty_threshold", type=float, default=3.0,
        help="Threshold for avg distance in novelty to be added to novelty queue.")
    parser.add_argument("--novelty_queue_size", type=int, default=1000,
        help="Number of novelty final states in the queue.")
    parser.add_argument("--novelty_neighbors", type=int, default=15,
        help="Number of novelty neighbors used to compute novelty.")


    parser.add_argument("--save_file", default="models/population.json",
        help="Directory to save NEAT models.")
    parser.add_argument("--load", action="store_true",
        help="Load existing population from save_file.")

    args = parser.parse_args()
    main(args)