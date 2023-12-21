import numpy as np
from numpy.random import default_rng

# assign rng
rng = default_rng()

# name of bees
# We added these names for debugging purposes
names = ["Patricia",
         "Jaqueline",
         "Georgia",
         "Sylvie",
         "Aurore",
         "Valentine",
         "Sandrine",
         "Catherine",
         "Véronique",
         "Marie-Paule",
         "Annelise",
         "Olga",
         "Édmée",
         "Anahide",
         "Zoé"]


# This function by default searches for the maximum of the objective function
# Use the opposite of the function if you are searching for the minimum
def bee_search(obj_func,
               search_space,
               minimize=False,
               n_bees=10,
               n_workers=None,
               n_scouts=1,
               max_iter=1000,
               limit=5,
               step=1):
    # The minimize parameter indicates whether or not the objective function should be minimized, 
    # in which case it should maximize the opposite, or not.
    if minimize:
        def fitness(input_vect):
            return -1 * obj_func(input_vect)
    else:
        def fitness(input_vect):
            return obj_func(input_vect)

    # Some value and parameter checking
    if search_space.shape[1] != 2 and search_space.shape[0] != 2:
        raise ValueError(
            "search_space is of wrong shape, it should have a shape of (*,2) or (2,*), \
            reshape or change search space")

    # initializing the number of worker bees, if None
    if n_workers is None:
        n_workers = n_bees // 2

    # Each worker should be assigned a food source
    # So the number of food source is equal to the number of workers
    n_foods = n_workers

    # initialize the hive
    hive = Hive({
        'scouts': [Scout(fitness, name=rng.choice(names)) for _ in range(0, n_scouts)],
        'onlookers':
            [Onlooker(fitness, name=rng.choice(names)) for _ in range(0, n_bees - n_workers)],
        'workers': [Worker(fitness, name=rng.choice(names)) for _ in range(0, n_workers)]
    })

    # generate an array of random coordinates in the search space
    # of shape [[...] * food_sources_initial]
    initial_foods = generate_new_food(search_space, n_foods, quantity=limit)

    # Send all workers at the initial food
    for worker, food in zip(hive.get_workers(), initial_foods):
        worker.go_to_food(food)
        worker.dance()

    # Get the initial best food
    best_quality_food = max(hive.get_workers(), key=lambda worker_bee: worker_bee.food.quality).food
    
    # Algorithm loop
    for i in range(max_iter):
        """Workers phase"""
        
        for worker in hive.get_workers():
            # We first check if the source has been depleted by the onlookers
            # Indeed we gave the ability for onlooker to bring food to the hive, 
            # so the food source gets depleted by both type of bees.
            # Because of the loop, if we have a food source of quantity value 0,
            # the worker bee has to leave and join the scouts to search for a new food source.
            if worker.should_leave():
                new_scout = worker.leave_food_point()
                hive.get_scouts().append(new_scout)
                hive.get_workers().remove(worker)
                continue
                
            worker.dance() # Workers register their food source and give them a food source quality value            
            
            # Search for a better food source around the current food source
            new_solution = worker.look_around(step)
            # Evaluate food source
            new_solution_evaluation = worker.calculate_nectar(new_solution)
            # If food source is better, discard old food source and select new open
            # Else bring food to the hive, or leave if no more food is available
            if new_solution_evaluation > worker.food.quality:
                worker.food = new_solution
                worker.dance()
                if new_solution_evaluation > best_quality_food.quality:
                    best_quality_food = new_solution
            else:
                worker.bring_food()
                if worker.should_leave():
                    new_scout = worker.leave_food_point()
                    hive.get_scouts().append(new_scout)
                    hive.get_workers().remove(worker)

        """Onlookers phase"""
        # Onlookers choose a food source, among the worker ones.
        # This choice is made by probabilities, 
        # otherwise they'd just choose the best one and never explore around the other ones.
        
        # Calculate the probability value of the sources
        # with which they are preferred by the onlooker bees
        # First they watch the bees dance
        # choreography is of type (worker, worker.food, food.quality)
        choreography = np.array([worker.dance() for worker in hive.get_workers()])
        # print([c.quality for c in choreography[:, 0]])
        choreography[:, 1] = np.abs(choreography[:, 1])
        # calculate the sum of all the qualities
        sum_dances = np.sum(choreography[:, 1])
        # for functions that are flat on most of their definition space, sum could be zero
        # we have to check for that condition to prevent having nan values in the probabilities list
        if sum_dances == 0.0:
            probabilities = None
        else:
            # divide each value to get the probability for the onlooker to choose that food source
            probabilities = np.divide(choreography[:, 1], sum_dances).astype('float64')
            # In that case, every food source has the same probability of being chosen which is not optimal.
            # We want to have most likeliness for the most promising source

        for onlooker in hive.get_onlookers():
            # Onlookers choose a food source, then dance.
            onlooker.choose_preferred_source(choreography[:, 0], probabilities)
            onlooker.dance()
            # Onlookers search for a new food source
            new_solution = onlooker.look_around(step)
            # Evaluate food source
            new_solution_evaluation = onlooker.calculate_nectar(new_solution)
            # If food source is better, discard old food source and select new food source
            if new_solution_evaluation > onlooker.food.quality:
                onlooker.food = new_solution
                onlooker.dance()
                if new_solution_evaluation > best_quality_food.quality:
                    best_quality_food = new_solution
            # Else increase limit counter
            else:
                if onlooker.food.has_food():
                    onlooker.bring_food()
                # If the food is exhausted, leave the food source
                if onlooker.should_leave():
                    onlooker.leave_food_point()

        """Scouts phase"""
        # Scouts search for a new food source around the search space. 
        for scout in hive.get_scouts():
            # Find a new food source
            scout.find_new_food(search_space, limit)
            # Convert back to worker and go on that food source
            new_worker = scout.convert_worker(fitness)
            hive.get_scouts().remove(scout)
            hive.get_workers().append(new_worker)

        # We made that function to be a python generator
        # As shown in the notebook, the function returns the iteration and the best quality food at each iteration
        yield i, best_quality_food


class Hive:

    def __init__(self, bees):
        self.bees = bees

    def get_workers(self):
        return self.bees['workers']

    def get_onlookers(self):
        return self.bees['onlookers']

    def get_scouts(self):
        return self.bees['scouts']

    def get_unemployed(self):
        return self.bees['scouts'] + self.bees['onlookers']

    def get_all_bees(self):
        return self.get_unemployed() + self.bees['workers']

    def collective_memory(self):
        map(lambda bee: (bee.dance(), bee.food),
            self.get_scouts() + self.get_workers())


class Bee:
    # job is the bee's job, either onlooker, employee, or scout
    # evaluate is the value of the food location by the objective function of the alg, see the
    # food_source is the food source coordinates on which the flower is located
    # it is the possible solution it represents
    def __init__(self, evaluate, food=None, name=None):
        self.food = food
        self.name = name
        self._evaluate = evaluate

    def go_to_food(self, food):
        self.food = food

    def dance(self):
        if self.food.quality is None:
            quality = self.calculate_nectar(self.food)
            self.food.quality = quality
        return self.food, self.food.quality

    def calculate_nectar(self, food=None):
        if food is None:
            return self._evaluate(self.food.location)
        else:
            return self._evaluate(food.location)


class EmployedBee(Bee):

    def should_leave(self):
        return self.food.is_exhausted()

    def bring_food(self):
        self.food.quantity -= 1

    def look_around(self, step=1):
        new_location = single(self.food, step)
        new_food = Food(new_location)
        return new_food


class Worker(EmployedBee):

    def leave_food_point(self):
        self.food = None
        return Scout(self._evaluate, name=self.name)


class Onlooker(EmployedBee):

    def leave_food_point(self):
        self.food = None

    def choose_preferred_source(self, choreography, probabilities=None):
        self.go_to_food(rng.choice(choreography, p=probabilities))


class Scout(Bee):

    def find_new_food(self, ranges, quantity=1):
        new_food = generate_new_food(ranges, 1, quantity)[0]
        self.food = new_food

    def convert_worker(self, evaluate):
        return Worker(evaluate, food=self.food, name=self.name)


# generate an array of random coordinates in the search space of shape  (*,food_sources_initial)
def generate_new_food(food_source_ranges, number_of_food_sources, quantity=1):
    start, end = food_source_ranges
    foods = np.array([
        Food(rng.uniform(start, end), quantity=quantity) for _ in range(number_of_food_sources)
    ])
    return foods


def single(original_food, search_step):
    ivar = rng.integers(original_food.location.shape[0])
    new_location = original_food.location.copy()
    new_location[ivar] += new_location[ivar] + search_step * rng.uniform(-0.5, 0.5)
    return new_location


class Food:

    def __init__(self, location, quality=None, quantity=0.0):
        self.location = location
        # quality will be determined upon dancing
        self.quality = quality
        self.quantity = quantity

    def has_food(self):
        return self.quantity > 0

    def is_exhausted(self):
        return self.quantity == 0
