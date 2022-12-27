import numpy as np
from dataclasses import dataclass
from neat.organism import Organism

@dataclass
class RepoOrgansim:
    org: Organism
    novelty_score: float
    skill_descriptor: list

@dataclass
class Neighbor:
    repo_org: RepoOrgansim
    dist: float
    idx: int

class DynamicArchive:
    """Measures the novelty-based on the final state of an organism.""" 
    def __init__(self, config):
        self.config = config
        # List of novel final states 
        self.novel_archive = []
        self.org_id_set = set()

    def attempt_add_archive(self, org, skill_descriptor):
        # Don't readd organsims already in the archive
        if org.id in self.org_id_set:
            return

        if len(self.novel_archive) <= self.config.min_archive_size:
            # Add organsim if the archive size isn't big enough yet
            self.novel_archive.append(
                RepoOrgansim(org, self.config.novel_threshold, skill_descriptor))
            self.org_id_set.add(org.id)
        else:
            dists = []
            for i, repo_org in enumerate(self.novel_archive):
                dists.append(
                    Neighbor(
                        repo_org,
                        self._dist(skill_descriptor, repo_org.skill_descriptor),
                        i))
            
            dists = sorted(dists, key=lambda x: x.dist)


            novelty_score = self.avg_dist(dists)

            if dists[0].dist >= self.config.novel_threshold:
                added_repo_org = RepoOrgansim(org, novelty_score, skill_descriptor.copy())
                self.novel_archive.append(added_repo_org)
                self.org_id_set.add(org.id)

            elif dists[0].dist < self.config.novel_threshold and dists[1].dist >= self.config.novel_threshold:
                if dists[0].repo_org.novelty_score < novelty_score or dists[0].repo_org.org.avg_fitness < org.avg_fitness:
                    # Remove the other from the archive
                    del self.novel_archive[dists[0].idx]
                    self.org_id_set.remove(dists[0].repo_org.org.id)

                    # Add the new one to the archive
                    added_repo_org = RepoOrgansim(org, novelty_score, skill_descriptor.copy())
                    self.novel_archive.append(added_repo_org)
                    self.org_id_set.add(org.id)

            # if novelty_score >= self.config.novel_threshold:
            #     added_repo_org = RepoOrgansim(org, novelty_score, skill_descriptor.copy())
            #     self.novel_archive.append(added_repo_org)
            #     self.org_id_set.add(org.id)
            # elif dists[0].repo_org.org.avg_fitness < org.avg_fitness:
            #     # Remove the other from the archive
            #     del self.novel_archive[dists[0].idx]
            #     self.org_id_set.remove(dists[0].repo_org.org.id)

            #     # Add the new one to the archive
            #     added_repo_org = RepoOrgansim(org, novelty_score, skill_descriptor.copy())
            #     self.novel_archive.append(added_repo_org)
            #     self.org_id_set.add(org.id)


    def avg_dist(self, dists):
        """Get the average distance for novelty score assuming dists is sorted."""
        dist_sum = 0
        num_neighbors = min(self.config.novelty_neighbors, len(dists))

        for i in range(num_neighbors):
            dist_sum += dists[i].dist

        return dist_sum / num_neighbors

    def _dist(self, state_1, state_2):
        sum_sd = 0
        # Take the squared difference between all the elements in the state
        for i in range(len(state_1)):
            sum_sd += (state_1[i] - state_2[i]) ** 2
        
        # Return the L2 distance
        return sum_sd ** 0.5
    
    def get_orgs(self):
        return [o.org for o in self.novel_archive]
    
    def reset(self):
        self.novel_archive = []
        self.org_id_set = set()