import math, os, random
import numpy as np

from neat.invocation_counter import InvocationCounter
from neat.mutator import Mutator
from neat.species import Species
from neat.organism import Organism
from neat.reproduction import Reproduction
from neat.stagnation import Stagnation
from neat_dynamics.novelty.dynamic_qd import DynamicArchive

class DynamicPopulation:
    """Maintains the population of organisms."""
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.inv_counter = InvocationCounter()
        self.mutator = Mutator(self.args, self.inv_counter)
        self.breeder = Reproduction(self.args)
        self.cur_id = 1
        self.species_list = [] # List of different species
        self.stagnation = Stagnation(self.args)
        self.orgs = []
        self.generation = 0
        self.dynamic_archive = DynamicArchive(self.config)


    def setup(self, net):
        self.base_org = Organism(self.args, net)
        self.inv_counter.gid_counter = len(net.nodes)
        self.orgs = self.spawn(self.base_org, self.args.init_pop_size)
        self.speciate()

    def spawn(self, base_org, pop_size):
        """Spawn the initial population."""
        
        orgs = []
        
        for i in range(pop_size):
            copy_org = base_org.copy(self.cur_id) # Create a copy
            self.cur_id += 1
            prev_rand = self.args.mutate_link_weight_rand_rate
            self.args.mutate_link_weight_rand_rate = 1.0

            self.mutator.mutate_link_weights(copy_org.net) # Randomize the link weights
            self.args.mutate_link_weight_rand_rate = prev_rand
            orgs.append(copy_org) 
        
        return orgs

    def add_species(self, species):
        self.species_list.append(species)
        self.stagnation.add_species(species)

    def speciate(self):
        """Put the organisms into different species."""
        # Put the rest of the organisms in a species
        cur_org_idx = 0
        for i in range(self.args.max_species):
            num_org = max(self.args.init_pop_size//self.args.max_species, 1)
            cur_species = Species(self.args, len(self.species_list))
            self.add_species(cur_species)
            for _ in range(num_org):
                cur_species.add(self.orgs[cur_org_idx])
                cur_org_idx += 1
    
    def respeciate(self):
        """Respeciate the population into different species that better match."""
        retained_orgs = set()
        # Keep few best orgs in each species 
        for species in self.species_list:
            species.orgs = species.orgs[:self.args.respeciate_size]
            for org in species.orgs:
                retained_orgs.add(org.id)

        # Find best matchgin species for each organism 
        for org in self.orgs:
            if org.id not in retained_orgs:
                random.shuffle(self.species_list)
                min_species_val = None
                best_species = None
                for i, cur_species in enumerate(self.species_list):
                    speciate_val = self.speciate_fn(org.net, cur_species.first().net)

                    if best_species == None or speciate_val < min_species_val:
                        min_species_val = speciate_val
                        best_species = cur_species
                
                best_species.add(org)

    def speciate_fn(self, net_1, net_2):
        """Compare two networks to determine if they should form a new species."""
        num_disjoint = 0
        num_shared = 0
        trait_diff = 0
        for link_gid in net_1.links:
            if link_gid not in net_2.links:
                num_disjoint += 1
            else:
                num_shared += 1
                trait_diff += net_1.links[link_gid].trait.distance(net_2.links[link_gid].trait)

        for link_gid in net_2.links:
            if link_gid not in net_1.links:
                num_disjoint += 1
        
        if num_shared == 0:
            num_shared = 1
        return self.args.speciate_disjoint_factor * num_disjoint + self.args.speciate_weight_factor * (trait_diff/num_shared)
    
    def evolve(self, skill_descriptors: list):
        """Evolve the population using Dynamics-Aware Quality-Diversity for Efficient Learning of Skill Repertoires."""
        assert len(skill_descriptors) == len(self.orgs)

        self.generation += 1
        #self.dynamic_archive.reset()
        skill_desscriptors, self.orgs = zip(*sorted(zip(skill_descriptors, self.orgs), key= lambda x:x[1].avg_fitness, reverse=True))
        
        for i in range(len(skill_descriptors)):
            self.dynamic_archive.attempt_add_archive(
                self.orgs[i],
                skill_descriptors[i])
        
        self.orgs = self.dynamic_archive.get_orgs()
        num_reproduce = max(self.args.init_pop_size - len(self.orgs), self.config.min_reproduce)
        print("Reproducing ", num_reproduce)
        new_orgs = []
        
        min_fitness = min([org.avg_fitness for org in self.orgs])
        max_fitness = max([org.avg_fitness for org in self.orgs])
        
        if min_fitness == max_fitness:
            min_fitness -= 0.01
        
        fitness_sum = 0
        for org in self.orgs:
            org.age += 1
            org.adj_fitness = (org.avg_fitness - min_fitness) / (max_fitness - min_fitness)
            fitness_sum += org.adj_fitness
        
        fitness_probs = []
        for org in self.orgs:
            fitness_probs.append(org.adj_fitness / fitness_sum)
        
        for i in range(num_reproduce):
            parent_1 = np.random.choice(self.orgs, p=fitness_probs)
            parent_2 = np.random.choice(self.orgs, p=fitness_probs)

            child_net = self.breeder.reproduce_directional(
                parent_1.net, parent_2.net, parent_1.avg_fitness, parent_2.avg_fitness)

            self.mutate_child(child_net)
            created_org = Organism(self.args, child_net, gen=max(parent_1.generation, parent_2.generation) + 1, id=self.cur_id) 
            self.species_list[0].add(
               created_org)
            new_orgs.append(created_org)
            
            # Increment the current organism ID
            self.cur_id += 1
        
        self.orgs = self.orgs + new_orgs
        print("len(self.orgs)", len(self.orgs))


    def mutate_child(self, child_net):
        if random.random() <= self.args.mutate_add_node_rate:
            self.mutator.mutate_add_node(child_net)
        
        if random.random() <= self.args.mutate_add_link_rate:
            self.mutator.mutate_add_link(child_net)
        
        if random.random() <= self.args.mutate_link_weight_rate:
            self.mutator.mutate_link_weights(child_net)

    def reset(self):
        """Reset all the organsisms in the population."""
        for org in self.orgs:
            org.reset()



        
            