import random
import copy
import math
import time
import numpy as np

import tracemalloc

# Set to be covered
#universe = set(range(1, 11))

class SetsData:
    universe = []
#
    subsets = []

    def write_data(self, file_name='sets.txt', set_size=400, subsets_count=50, subsets_size=72):
        self.universe = set(range(1, set_size))
        for i in range(0, subsets_count):
            my_set = set(random.sample(range(1, set_size), subsets_size))
            self.subsets.append(my_set)

        with open(file_name, 'w') as filehandle:
            filehandle.write('%s\n' % self.universe)
            for i in range(0, subsets_count):
                filehandle.write('%s\n' % self.subsets[i])

    def read_data(self, file_name='sets.txt'):

        with open(file_name, 'r') as filehandle:
            lines = filehandle.read().splitlines()
            self.universe = set(int(item) for item in lines[0][1:-1].split(','))
            for i in range(0, len(lines) - 1):
                subset = set(int(item) for item in lines[i + 1][1:-1].split(','))
                self.subsets.append(subset)

# Set of subsets to cover the universe
#subsets = [
#    set([1, 3, 4]),
#    set([2, 3, 4, 5]),
#    set([4, 5, 6]),
#    set([6, 7, 9]),
#    set([8, 9, 10]),
#    set([2, 9, 10]),
#    set([1, 5, 9]),
#    set([3, 4])
#]

class Cat:
    def __init__(self, sets_data):
        self.position = [bool(random.getrandbits(1)) for _ in range(len(sets_data.subsets))]
        self.velocity0 = np.zeros(len(sets_data.subsets))
        self.velocity1 = np.zeros(len(sets_data.subsets))
        self.fitness = 0
        for i in range(len(sets_data.subsets)):
            if self.position[i] == 1:
                self.fitness += 1#len(sets_data.subsets[i])
        for elem in sets_data.universe:
            is_elem_covered = False
            for i in range(len(sets_data.subsets)):
                if self.position[i] == 1:
                    if elem in sets_data.subsets[i]:
                        is_elem_covered = True
                        break
            if not is_elem_covered:
                self.fitness = np.inf
                break

    def evaluate_fitness(self, sets_data):
        self.fitness = 0
        for i in range(len(sets_data.subsets)):
            if self.position[i] == 1:
                self.fitness += 1#len(sets_data.subsets[i])
        for elem in sets_data.universe:
            is_elem_covered = False
            for i in range(len(sets_data.subsets)):
                if self.position[i] == 1:
                    if elem in sets_data.subsets[i]:
                        is_elem_covered = True
                        break
            if not is_elem_covered:
                self.fitness = np.inf
                break


class CatSwarmOptimizer:

    def __init__(self, num_cats, max_iterations, sets_data):
        self.sets_data = sets_data
        self.num_cats = num_cats
        self.max_iterations = max_iterations
        self.cats = [Cat(self.sets_data) for _ in range(num_cats)]
        self.mixture_ratio = 0.5

    def tracing(self, trace_cat, best_cat):
        c1 = 1
        r1 = random.random()
        w = 1
        Vmax = 0.7

        for d in range(len(self.sets_data.subsets)):
            # step1
            if best_cat.position[d]:
                d1d = r1 * c1
                d0d = -r1 * c1
            else:
                d1d = -r1 * c1
                d0d = r1 * c1
            # step2
            trace_cat.velocity1[d] = max(min(w * trace_cat.velocity1[d] + d1d, Vmax), 0)
            trace_cat.velocity0[d] = max(min(w * trace_cat.velocity0[d] + d0d, Vmax), 0)
            # step3
            if trace_cat.position[d]:
                Vd = trace_cat.velocity0[d]
            else:
                Vd = trace_cat.velocity1[d]
            # step4
            td = 1 / (1 + math.exp(-Vd))
            # step5
            if random.random() < td:
                trace_cat.position[d] = best_cat.position[d]

        trace_cat.evaluate_fitness(self.sets_data)

        return trace_cat.position, trace_cat.fitness, trace_cat

    def seeking(self, seek_cat):
        PMO = 0.7  # (Probability of Mutation Operation)
        CDC = random.randrange(len(self.sets_data.subsets) + 1)  # (Counts of Dimensions to Change)
        SMP = [Cat(self.sets_data) for _ in range(30)]  # (Seeking Memory Pool)

        # step1
        local_best_position = seek_cat.position
        local_best_fitness = seek_cat.fitness
        for i in range(len(SMP)):
            SMP[i] = copy.deepcopy(seek_cat)

        # step2
        count = random.sample(range(len(self.sets_data.subsets)), CDC)
        for i in range(len(SMP)):
            for k in range(len(self.sets_data.subsets)):
                flag = False
                for j in range(len(count)):
                    if k == count[j]:
                        flag = True
                if flag == True:
                    if random.random() < PMO:
                        if SMP[i].position[k]:
                            SMP[i].position[k] = False
                        else:
                            SMP[i].position[k] = True

        # step3
        minF = np.inf
        maxF = 0
        for i in range(len(SMP)):
            SMP[i].evaluate_fitness(self.sets_data)
            if SMP[i].fitness < minF:
                minF = SMP[i].fitness
            if SMP[i].fitness > maxF:
                if SMP[i].fitness != np.inf:
                    maxF = SMP[i].fitness
            if SMP[i].fitness < local_best_fitness:
                local_best_position = SMP[i].position
                local_best_fitness = SMP[i].fitness

        if minF == np.inf:
            return local_best_position, local_best_fitness, seek_cat

        # step4
        Pi = [0 for _ in range(len(SMP))]
        if minF == maxF:
            for i in range(len(SMP)):
                if SMP[i].fitness == maxF:
                    Pi[i] = 1
        else:
            for i in range(len(SMP)):
                if SMP[i].fitness != np.inf:
                    Pi[i] = (maxF - SMP[i].fitness) / (maxF - minF)

        # step5
        res = random.choices(SMP, weights=Pi, k=1)

        # step6
        return local_best_position, local_best_fitness, res[0]

    def optimize(self):
        global_best_cat = Cat(self.sets_data)

        for n in range(self.max_iterations):
            # Update mixture ratio
            num_tracking_cats = int(self.mixture_ratio * self.num_cats)
            count = random.sample(range(len(self.cats)), num_tracking_cats)

            for i in range(len(self.cats)):
                flag = False
                for j in range(len(count)):
                    if i == count[j]:
                        flag = True
                if flag == True:
                    l_b_position, l_b_fitness, l_cat = self.tracing(self.cats[i], global_best_cat)
                    self.cats[i] = l_cat
                    if (global_best_cat.fitness > l_b_fitness):
                        global_best_cat.position = l_b_position
                        global_best_cat.fitness = l_b_fitness
                else:
                    l_b_position, l_b_fitness, l_cat = self.seeking(self.cats[i])
                    self.cats[i] = l_cat
                    if (global_best_cat.fitness > l_b_fitness):
                        global_best_cat.position = l_b_position
                        global_best_cat.fitness = l_b_fitness


            # Update mixture ratio
            #self.mixture_ratio = 0.9 * self.mixture_ratio + 0.1 * (global_best_fitness / np.mean([cat.fitness for cat in self.cats]))

        return global_best_cat.position, global_best_cat.fitness

if __name__ == '__main__':
    start = time.time()

    tracemalloc.start()

    sets_data = SetsData()
    #sets_data.write_data("sets.txt", 400, 50, 72)
    sets_data.read_data()
    csa = CatSwarmOptimizer(30, 10, sets_data)
    best_position, best_fitness = csa.optimize()
    end = time.time() - start
    print(end)
    print("Best position:", best_position)
    print("Best fitness:", best_fitness)

    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics('lineno')

    total = sum(stat.size for stat in stats)
    print(f"Total allocated memory: {total} bytes")
