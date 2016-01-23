
class Distance(object):
    EUCLIDIAN_DISTANCE = 1
    MAHALANOBIS_DISTANCE = 2
    
    @staticmethod
    def get_euclidian_distance(training_example, test_example):
            import math
            return math.sqrt((training_example.x - test_example.x)**2 +\
                             (training_example.y - test_example.y)**2)
        
    @staticmethod
    def get_mahalanobis_distance(training_example, test_example):
        import math
        return math.sqrt(((training_example.x - test_example.x)**2) / training_example.standard_deviation**2 +\
                         ((training_example.y - test_example.y)**2) / training_example.standard_deviation**2)
    
    @staticmethod
    def get_distance_calculator(distance_type):
        
        d = {Distance.EUCLIDIAN_DISTANCE: Distance.get_euclidian_distance,
             Distance.MAHALANOBIS_DISTANCE: Distance.get_mahalanobis_distance
            }
        return d[distance_type]

class Example(object):
    def __init__(self, x, y, value, klass=""): 
        self.x = x
        self.y = y
        self.value = value
        self.weight = 0.0
        self.distance = 0.0
        self.distance_type = 0
        self.klass = klass
        self.standard_deviation = 0.0

    def get_value(self):
        return self.value
    
    def get_location(self):
        return (self.x, self.y)
    
    def get_weight(self):
        return self.weight
    
    def set_weight(self, weight):
        self.weight = weight

    def get_distance(self):
        return self.distance
    
    def set_distance(self, distance, distance_type=Distance.EUCLIDIAN_DISTANCE):
        self.distance_type = distance_type
        self.distance = distance
    
    def get_distance_type(self):
        return self.distance_type
    
    def get_class(self):
        return self.klass
    
    def set_class(self, klass):
        self.klass = klass
        
    def set_standard_deviation(self, standard_deviation):
        self.standard_deviation = standard_deviation
    
    def get_standard_deviation(self, standard_deviation):
        return self.standard_deviation
    
        
class TestExample(Example):
    pass

class TrainingExample(Example):   
    pass

class Knn(object):
    def __init__(self, k):
        self.k = k
        self.training_examples = []
        self.test_example = None
        
    def choose_k(self):
        pass
    
    def get_mean(self, training_examples, test_example):
        dataset = training_examples[:]
        dataset.append(test_example)
        sum = 0.0
        for t_e in dataset:
            sum += t_e.value
        return sum/len(dataset)
    
    def get_standard_deviation(self, training_examples, test_example):
        import math
        dataset = training_examples[:]
        dataset.append(test_example)
        mean = self.get_mean(training_examples, test_example)
        sum = 0.0
        for e in dataset:
            sum += (e.value - mean)**2
        return math.sqrt(float(sum)/len(dataset))
    
    def set_dataset_standard_deviation(self):
        standard_deviation = self.get_standard_deviation(self.training_examples, self.test_example)
        dataset = self.training_examples[:]
        dataset.append(self.test_example)
        
        for e in dataset:
            e.standard_deviation = standard_deviation

    def set_training_examples_distances_weight(self, distance_type):
        distance_calculator = Distance.get_distance_calculator(distance_type)
        for training_example in self.training_examples:
            d = distance_calculator(training_example, self.test_example)
            training_example.set_distance(d, distance_type)
            training_example.set_weight(1.0/d if d else 10**9)
    
    def get_k_neighbors(self):
        from operator import attrgetter
        self.training_examples.sort(key=attrgetter('weight'), reverse=True)
        return self.training_examples[:self.k]
    
    def get_test_example_class(self):
        k_neighbors = self.get_k_neighbors()
        classes_weights = {}
        
        for n in k_neighbors:
            klass = n.get_class()
            weight = n.get_weight()
            classes_weights[klass] = classes_weights.get(klass, 0) + weight
            
        result_class_value  = (None, 0) # optimize since 0 may be > all -ve values
                
        for k, v in classes_weights.iteritems():
            if v > result_class_value[1]:
                result_class_value = (k, v)
        
        return result_class_value[0]


class UseCase(object):
    # grade A [80, 90, 100]
    # grade B [50, 60, 70]
    # grade C [20, 30, 40]
    
    def run(self):
        training_examples =  [TrainingExample(2, 0, 2, "d"),
                              TrainingExample(3, 0, 3, "c"),
                              TrainingExample(4, 0, 4, "c"),
                              TrainingExample(20, 0, 20, "C"),
                              TrainingExample(30, 0, 30, "C"),
                              TrainingExample(40, 0, 40, "C"),
                              TrainingExample(50, 0, 50, "B"),
                              TrainingExample(60, 0, 60, "B"),
                              TrainingExample(70, 0, 70, "B"),
                              TrainingExample(80, 0, 80, "A"),
                              TrainingExample(90, 0, 90, "A"),
                              TrainingExample(100, 0, 100, "A")
                              ]
    
        test_example = TestExample(20, 0, 20)
        
        knn = Knn(10)
        knn.training_examples = training_examples
        knn.test_example = test_example
        knn.set_dataset_standard_deviation()
        knn.set_training_examples_distances_weight(Distance.MAHALANOBIS_DISTANCE)
        return knn.get_test_example_class()
    
    
if __name__ == "__main__":
    print UseCase().run()
    