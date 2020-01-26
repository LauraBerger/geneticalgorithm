import random

import numpy as np
import math

from colored import colored

from FitnessCalculation import Histogram, Fitness_Calc
import matplotlib.pyplot as plt
import time


class GA:

    def __init__(self, chromosomes_equal, population_size, base_image, trans_image):
        self.base_image = base_image
        self.trans_image = trans_image
        self.fitness_calc = Fitness_Calc(base_image=base_image, trans_image=trans_image)

        self.chromosomes_values_mask = ((-200, 200, 2), (0, 2, 100, 2), (-180, 180, 10, 1), (-45, 45, 10, 2))
        # self.chromosomes_values_mask = ((-250, 250, 2), (0, 2, 2), (0, 360, 1), (0, 0.0001, 2))

        # self.chromosomes_equal = chromosomes_equal

        self.population_size = population_size
        self.population = self.new_random_population(self.population_size)
        self.chromosomes = self.population.shape[1]

        # self.fitness_evaluation = fitness_evaluation
        self.fitness = self.calculate_fitness(self.population)
        self.p_age = np.zeros(population_size)

        # Mutationsrate der Childrens ( population_size/2 )
        self.offspring_mutation_rate = 0.1
        self.mutation_rate = self.offspring_mutation_rate

        self.max_generations = 500

        self.new_pop = False
        self.converged = False
        self.allow_convergences = 0.1
        self.allow_converged_cycles = 20
        self.converged_fitness = self.fitness
        self.converged_fitness_value_difference = 0.01
        self.convergence_counter = 0
        self.p_max=0
        self.p_max_co=0
        self.p_max_cy=200

        self.best_i = int(self.population_size/16)
        self.best_i_i = self.best_i

    def get_population(self):
        return self.population

    def set_new_population(self, population):
        self.population = population
        self.population_size = population.shape[0]
        self.fitness = self.calculate_fitness(population)

    def new_random_population(self, population_size):

        # population = np.hstack([np.random.uniform(low=x[0], high=x[1],
        #                                        size=[population_size, x[2]]) for x in self.chromosomes_values_mask])

        trans = np.random.randint(low=self.chromosomes_values_mask[0][0], high=self.chromosomes_values_mask[0][1],
                                  size=[population_size, self.chromosomes_values_mask[0][2]])
        scale = np.random.randint(low=self.chromosomes_values_mask[1][0], high=self.chromosomes_values_mask[1][1],
                                  size=[population_size, self.chromosomes_values_mask[1][3]]) + np.random.randint(low=0,
                                                high=self.chromosomes_values_mask[1][ 2],size=[population_size,
                                                                        self.chromosomes_values_mask[ 1][3]]) / \
                                                                              self.chromosomes_values_mask[1][2]
        rotation = np.random.randint(low=self.chromosomes_values_mask[2][0], high=self.chromosomes_values_mask[2][1],
                                     size=[population_size, self.chromosomes_values_mask[2][3]]) + np.random.randint(
            low=0, high=self.chromosomes_values_mask[2][2],
            size=[population_size, self.chromosomes_values_mask[2][3]]) / self.chromosomes_values_mask[2][2]
        shear = np.random.randint(low=self.chromosomes_values_mask[3][0], high=self.chromosomes_values_mask[3][1],
                                  size=[population_size, self.chromosomes_values_mask[3][3]]) + np.random.randint(low=0,
                                                                                                                  high=
                                                                                                                  self.chromosomes_values_mask[
                                                                                                                      3][
                                                                                                                      2],
                                                                                                                  size=[
                                                                                                                      population_size,
                                                                                                                      self.chromosomes_values_mask[
                                                                                                                          3][
                                                                                                                          3]]) / \
                self.chromosomes_values_mask[3][2]

        population = np.hstack((trans, scale, rotation, shear))

        return population

    def get_fitness(self):
        return self.fitness

    def calculate_fitness(self, population):
        population_size = population.shape[0]

        fitness = np.stack(
            [self.fitness_calc.fitness(population[x, :]) for x in range(0, population_size)], axis=0)
        #mutual_information = np.stack([self.fitness_calc.nmi_fitness(population[x,:]) for x in range(0, population_size)], axis=0)
        #print(mutual_information)
        return fitness

    def faster_fitness_calc(self, population):

        p_population, index = np.unique(self.population, axis=0, return_index=True)
        p_fitness = self.fitness[index]

        equal_index = np.where((population == p_population[:, None]).all(-1))

        temp_population = population[equal_index[1]]
        temp_fitness = p_fitness[equal_index[0]]
        calc_population1 = np.delete(population, equal_index[1], axis=0)

        if len(calc_population1) > 0:
            calc_fitness = self.calculate_fitness(calc_population1)
            fitness = np.concatenate((temp_fitness, calc_fitness))
            population = np.concatenate((temp_population, calc_population1), axis=0)
        else:
            fitness = self.fitness
            population = self.population

        self.fitness = fitness
        self.population = population

    def sort_by_fitness(self):
        # i = np.flip(np.argsort(self.fitness))
        i = np.argsort(self.fitness)
        self.population = self.population[i]
        self.fitness = self.fitness[i]

    # def convergence_test(self):
    #
    #     self.sort_by_fitness()
    #
    #     c_test = abs(self.fitness - self.converged_fitness) > self.converged_fitness_value_difference
    #     c_inv = 1.0 - (np.sum(c_test) / len(self.converged_fitness))
    #     # print(c_test, c_inv,self.convergence_counter,self.converged)
    #
    #     if c_inv >= self.allow_convergences and self.convergence_counter == self.allow_converged_cycles:
    #         self.converged = True
    #     elif c_inv >= self.allow_convergences:
    #         self.convergence_counter += 1
    #         # self.converged = False
    #
    #     else:
    #         self.convergence_counter = 0
    #         # self.converged = False
    #
    #     self.converged_fitness = self.fitness
    #
    #     # c_test = abs(self.fitness - self.fitness[-1]) > self.converged_fitness_value_difference
    #     # c_inv = 1.0 - (np.sum(c_test) / len(self.converged_fitness))
    #     # # print(c_test, c_inv,self.convergence_counter,self.converged)
    #     #
    #     # if c_inv >= self.allow_convergences and self.convergence_counter == self.allow_converged_cycles:
    #     #     self.converged = True
    #     # elif c_inv >= self.allow_convergences:
    #     #     self.convergence_counter += 1
    #     #     self.new_pop = True
    #     #
    #     # else:
    #     #     self.convergence_counter = 0
    #     #     self.new_pop = False
    #
    #     self.converged_fitness = self.fitness


    def convergence_test(self):
        # p_max = fitness[-1]
        # p_mean = np.median(fitness[int(len(fitness)/2):])
        # p_diff = p_max-p_mean
        # p_close= np.isclose(fitness[int(len(fitness)/2):], p_mean, atol=1e-03,rtol=1e-03)
        # #print("max ", p_max, "mean ", p_mean, "diff ", p_diff, "close ", p_close)
        # np_close = np.sum(p_close)/len(p_close)
        #
        # if np_close >= self.allow_convergences and self.convergence_counter == self.allow_converged_cycles:
        #     self.converged = True
        # elif np_close >= self.allow_convergences:
        #     print("ja")
        #     self.convergence_counter += 1
        #
        # else:
        #     self.convergence_counter = 0
        #     self.new_pop = False
        self.sort_by_fitness()
        p_max = self.fitness[-1]
        p_close = np.isclose(self.fitness, p_max, atol=1e-02, rtol=1e-02)
        #print("max ", p_max, "close ", p_close)
        np_close = np.sum(p_close) / len(p_close)


        if  p_max==self.p_max and self.p_max_co>=self.p_max_cy:
            self.converged=True
        elif p_max==self.p_max:
            self.p_max_co+=1
        else:
            self.p_max_co=0

        if np_close >= self.allow_convergences:
            self.convergence_counter += 1

        else:
            self.convergence_counter = 0

        self.p_max=p_max


    def dynamic_mutation_rate(self, generation_counter):
        # m = (self.max_generations-generation_counter)/self.max_generations
        # if m > 0.75:
        #     self.mutation_rate = 0.9
        # elif m<0.25:
        #     self.mutation_rate = 0.2
        # else:
        #     self.mutation_rate = m

        if self.convergence_counter > self.allow_converged_cycles / 2:
            self.mutation_rate = 0.75
            #self.best_i_i = int(self.best_i / 2)
        else:
            self.mutation_rate = self.offspring_mutation_rate
            self.best_i_i = self.best_i

    def genetic_algorithm_cycle(self):

        generation_counter = 0
        generation_start_time = time.clock()

        while (generation_counter < self.max_generations and not self.converged):
            start = time.clock()
            population = self.population_cycle()
            end1 = time.clock()
            # self.set_new_population(population)
            self.faster_fitness_calc(population)
            end2 = time.clock()

            #print("mut: ", self.mutation_rate)

            end3 = time.clock()

            # print(self.population, "\n", self.fitness)
            self.convergence_test()
            self.dynamic_mutation_rate(generation_counter)


           #print(self.fitness, "\n")
            print()
            print()
            print()
            print(generation_counter)
            print()
            print()
            #print(colored(generation_counter,'red'))

            #input("Press Enter to continue...")

            #print("pop: ", end1 - start, " fit: ", end2 - end1, " conv: ", end3 - end2)
            generation_counter += 1

        generation_end_time = time.clock()
        generation_time = generation_end_time - generation_start_time

        if self.converged:
            print("\033[93m" + 'population has converged!' + "\033[0m")
        else:
            print("\033[93m" + 'generation limit has been reached!' + "\033[0m")

        print("\033[93m" + 'solution after: ' + str(generation_counter) + " generations" + " & " + str(
            generation_time) + " sec." + "\033[0m")
        # print(self.population, "\n", self.fitness)

    def population_cycle(self):
        selected_population = self.selection()
        crossover_population = self.crossover(selected_population)
        mutation_population = self.mutation(crossover_population)

        return np.concatenate((selected_population, mutation_population), axis=0)

    def best_solution(self):
        i = np.argmax(self.fitness)
        return self.population[i], self.fitness[i]

    def select_individual_by_tournament(self,population, scores):
        # es werden zwei zufällige Positionen bestimmt an denen der Fitnesswert ermittelt wird
        # einer von beiden hat immer eine höhere Fitness
        # alle Gewinner werden in einem Array zurückgegeben
        population_size = len(scores)  # was ist die populations größe?
        fighter_1 = random.randint(0, population_size - 1)
        fighter_2 = random.randint(0, population_size - 1)
        fighter_1_fitness = scores[fighter_1]
        fighter_2_fitness = scores[fighter_2]
        if fighter_1_fitness >= fighter_2_fitness:
            winner = fighter_1
        else:
            winner = fighter_2
        # print(population[winner,:])
        return (population[winner,:])

    def get_probability_list(self,population_scores):
        fitness = population_scores.values()
        total_fit = float(sum(fitness))
        relative_fitness = [f / total_fit for f in fitness]
        probabilities = [sum(relative_fitness[:i + 1])
                         for i in range(len(relative_fitness))]
        return probabilities
    def roulette_wheel_pop(self,population, scores):
        start = time.time()
        chosen = []
        array = np.arange(0.5, 101.1, 1, int)
        population_scores = dict(zip(array, scores))
        print(array)
        probabilities = self.get_probability_list(population_scores)
        # for n in range(number):
        r = random.random()
        for (i, individual) in enumerate(population):
            if r <= probabilities[i]:
                chosen.append(individual)
                break
        end = time.time()
        # print("Gesamtzeit Selection: ",end - start)
        return chosen
    def selection(self):
        """Fittest selection"""
        # self.sort_by_fitness()
        # selected_population = self.population[-int(self.population_size/2):]


        """Roulette Wheel Selection"""
        # p_selection = self.fitness / np.sum(self.fitness)
        # selection = np.random.choice(np.arange(0, self.population.shape[0]),
        #                              size= int (self.population_size / 2), replace=True, p=p_selection)
        # selected_population = self.population[selection, :]
        #selected_population=self.roulette_wheel_pop(self.population,self.fitness)

        """Mix: Elitism Selection & Roulette Wheel Selection"""
        # p_population, index = np.unique(self.population, axis=0, return_index=True)
        # p_fitness = self.fitness[index]
        #
        # i = np.argsort(p_fitness)
        # p_fitness = p_fitness[i]
        # p_population = p_population[i]
        #
        # best_i = self.best_i_i
        #
        # p_selection = p_fitness / np.sum(p_fitness)
        # selection = np.random.choice(np.arange(0, p_population.shape[0]), size=int(self.population_size / 2) - best_i,
        #                                  replace=True, p=p_selection)
        # selected_population = p_population[selection, :]
        #
        # if best_i > 0:
        #     selected_population = np.concatenate((selected_population, p_population[-best_i:]), axis=0)
        #
        """Tournament"""
       # p_population, index = np.unique(self.population, axis=0, return_index=True)
       # p_fitness = self.fitness[index]
       # selected_population = [0, 0, 0, 0, 0, 0, 0]
        #for i in range(len(p_population / 2)):
         #   parent = self.select_individual_by_tournament(p_population, p_fitness)
            # selected_population=selected_population[parent]
            # selected_population = s_population[parent, :]
            # selected_population.append(parent)
            # selected_population[i,:]=parent
          #  selected_population = np.vstack((selected_population, parent))
        #   selected_population=np.concatenate((selected_population,parent))
        #selected_population = np.delete(selected_population, 0, 0)


        """Rank Selection"""
        #print(colored('Population', 'red'), colored(self.population, 'green'))
        #print("Population",self.population)
        p_population, index = np.unique(self.population, axis=0, return_index=True)
        p_fitness = self.fitness[index]
        #print("index",index)
        #print(("p_fitness:  ",p_fitness))
        best_i = self.best_i_i

        s_population = p_population[:-best_i, :]

        # # p_population=self.population
        # # p_fitness = self.fitness


        i = np.argsort(p_fitness)
        p_fitness = p_fitness[i]
        p_population = p_population[i]

        best_i = self.best_i_i

        s_population = p_population[:-best_i,:]
        s_fitness = p_fitness[:-best_i]


        p_selection = np.arange(1, len(s_fitness) + 1, dtype=float) / np.sum(
            np.arange(1, len(s_fitness) + 1, dtype=float))
        selection = np.random.choice(np.arange(0, s_population.shape[0]), size=int(self.population_size / 2)-best_i,
                                     replace=True, p=p_selection)
        print("selection",selection)
        selected_population = s_population[selection, :]
        selected_population = np.concatenate((selected_population, p_population[-best_i:]), axis=0)


        # i = np.argsort(self.fitness)
        # p_fitness = self.population[i]
        # p_population = self.population[i]
        #
        # p_selection = np.arange(1, len(p_fitness) + 1, dtype=float) / np.sum(
        #     np.arange(1, len(p_fitness) + 1, dtype=float))
        # selection = np.random.choice(np.arange(0, p_population.shape[0]), size=int(self.population_size / 2),
        #                              replace=True, p=p_selection)
        # selected_population = p_population[selection, :]

        print("neue Population",selected_population)
        return selected_population

    def crossover(self, population):
        population_size = population.shape[0]
        cross_i_amount = (int)(population_size / 2)
        cross_individuals = np.random.choice(np.arange(0, population_size), size=(cross_i_amount, 2),
                                             replace=False)

        a = True
        cross_mask = ()

        while a:
            cross_mask = np.random.choice((0, 1), size=(cross_i_amount, self.chromosomes),
                                          replace=True, p=(0.5, 0.5))
            sum = np.sum(cross_mask, axis=1)
            if np.sum(sum == 0) == 0 and np.sum(sum == self.chromosomes) == 0:
                a = False
            else:
                a = True

        # cross_mask = np.tile((0,0,0,1,1,1,1),(cross_i_amount,1))

        inv_cross_mask = cross_mask ^ 1

        p1 = population[cross_individuals[:, 0], :]
        p2 = population[cross_individuals[:, 1], :]
        ch1 = p1 * cross_mask + p2 * inv_cross_mask
        ch2 = p2 * cross_mask + p1 * inv_cross_mask

        children_population = np.concatenate((ch1, ch2), axis=0)
        return children_population

    def mutation(self, population):
        mutation = np.random.choice((0, 1), size=population.shape, replace=True,
                                    p=(self.mutation_rate, 1 - self.mutation_rate))
        inv_mutation = mutation ^ 1
        mutation_value = self.new_random_population(population.shape[0])

        mutated_population = population * mutation + mutation_value * inv_mutation
        return mutated_population
