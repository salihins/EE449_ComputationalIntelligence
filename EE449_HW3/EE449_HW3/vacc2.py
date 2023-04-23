import numpy as np
import skfuzzy as fuzz
from vaccination import Vaccination
from skfuzzy import control as ctrl

# Create universal variables to use in Antecedent and Consequent functions
# vacc_per = np.linspace(0, 1.0, 0.05)
# sigma = np.linspace(-0.2, 0.2, 0.05)
# linspace caused the following error.
# TypeError: 'float' object cannot be interpreted as an integer
# Hence, it is replaced with np.arange which is taken from:
# https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem_newapi.html


#References: https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html#example-plot-tipping-problem-py
#https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem_newapi.html
#https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_control_system_advanced.html
#https://stackoverflow.com/questions/65596610/fuzzy-system-valueerror
#Some parts are taken directly from the above websites. Just changed the names.

#Class definition
class myFuzzy():
    def __init__(self):
        #Partition definitions are take directly from here
        #https://stackoverflow.com/questions/65596610/fuzzy-system-valueerror
        #https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem_newapi.html
        #Just adjusted the variable names and values
        self.vacc_set = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'vacc_set')
        self.failure = ctrl.Antecedent(np.arange(-1, 1.01, 0.01), 'failure')
        self.cont_rate = ctrl.Consequent(np.arange(-0.2, 0.21, 0.05), 'cont_rate')

        # Define the partitions for vaccination
        self.vacc_set['low'] = fuzz.trapmf(self.vacc_set.universe, [0, 0, 0.4, 0.6])
        self.vacc_set['avg'] = fuzz.trimf(self.vacc_set.universe, [0.4, 0.6, 0.8])
        self.vacc_set['high'] = fuzz.trapmf(self.vacc_set.universe, [0.6, 0.8, 1, 1])
        self.vacc_set.view()

        # Define the partitions for control rate
        self.cont_rate['high'] = fuzz.trapmf(self.cont_rate.universe, [0.05, 0.1, 0.2, 0.2])
        self.cont_rate['avghigh'] = fuzz.trimf(self.cont_rate.universe, [0, 0.05, 0.1])
        self.cont_rate['avg']= fuzz.trimf(self.cont_rate.universe, [-0.05, 0, 0.05])
        self.cont_rate['avglow'] = fuzz.trimf(self.cont_rate.universe, [-0.1, -0.05, 0])
        self.cont_rate['low']= fuzz.trapmf(self.cont_rate.universe, [-0.2, -0.2, -0.1, 0.05])
        self.cont_rate.view()

        #Define the partitions for failure
        self.failure['low'] = fuzz.trapmf(self.failure.universe, [-1, -1, -0.5, 0])
        self.failure['avg'] = fuzz.trimf(self.failure.universe, [-0.5, 0, 0.5])
        self.failure['high'] = fuzz.trapmf(self.failure.universe, [0, 0.5, 1, 1])
        self.failure.view()

        #Syntax for rules are taken from here
        #https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem_newapi.html

        #Rule1: If vaccination and failure rates are high, then control can be low.
        rule1 = ctrl.Rule(antecedent=(self.vacc_set['high'] & self.failure['high']), consequent=self.cont_rate['low'])
        # Rule2: Between vaccination and failure rates, if one of them is high and the other one is average, then control can be increased to average low.
        rule2 = ctrl.Rule(antecedent=((self.vacc_set['high'] & self.failure['avg']) | (self.vacc_set['avg'] & self.failure['high']) ),
                          consequent= self.cont_rate['avglow'])
        #Rule3: If the mean of the failure and vaccination rates is 'average', then control can be increased to average.
        #For example vaccination rate is low, failure rate is high -> (low+high)/2 = average
        rule3 = ctrl.Rule(antecedent=((self.vacc_set['avg'] & self.failure['avg']) | (self.vacc_set['low'] & self.failure['high']) |
                                      (self.vacc_set['high'] & self.failure['low'])), consequent=self.cont_rate['avg'])
        # Rule4: Between vaccination and failure rates, if one of them is low and the other one is average, then control can be increased to average high.
        rule4 = ctrl.Rule(antecedent=((self.vacc_set['low'] & self.failure['avg']) | (self.vacc_set['avg'] & self.failure['low'])),
                          consequent=self.cont_rate['avghigh'])
        # Rule5: If vaccination and failure rates are high, then control can be low.
        rule5 = ctrl.Rule(antecedent=(self.vacc_set['low'] & self.failure['low']), consequent=self.cont_rate['high'])

        #Set control rules
        self.cont_rule = ctrl.ControlSystem(rules=[rule1, rule2, rule3, rule4, rule5])
        #Call Vaccination.py
        self.model = Vaccination()


    def FuzzyLogic(self):
        #Directly taken from here
        #https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem_newapi.html
        self.fuzz = ctrl.ControlSystemSimulation(self.cont_rule)
        self.percentage = self.model.checkVaccinationStatus()[0]
        self.failure_inp = self.model.checkVaccinationStatus()[1]
        #Give inputs
        self.fuzz.input['vacc_set'] = self.percentage
        self.fuzz.input['failure'] = self.failure_inp
        #Compute the system
        self.fuzz.compute()
        #Get the output
        self.output_Control = self.fuzz.output['cont_rate']
        self.model.vaccinatePeople(self.output_Control)
    def UpdatePercent(self):
        #Update the variable
        self.vacc_perc = self.model.checkVaccinationStatus()[0]
    def UpdateFail(self):
        # Update the variable
        self.failure = self.model.checkVaccinationStatus()[1]
    def UpdateCost(self):
        # Update the variable
        self.cost = self.model.vaccination_rate_curve_[-1]

#Create a variable for check to eq. point.
checker = 1
#Initialize cost
cost = 0
#Define error value
error = 0.0001
#Call the fuzzy system
Pop = myFuzzy()
for i in range(200):
    #Initialize logic
    Pop.FuzzyLogic()
    #Update variables
    Pop.UpdatePercent()
    Pop.UpdateCost()
    Pop.UpdateFail()
    # Update variables
    if checker == 1:
        cost += Pop.cost #Update the cost until eq. point
    diff = abs(Pop.vacc_perc - 0.6) #Check the difference
    if (diff < error) and (checker == 1): #If eq. conditions are satisfied
        checker = 0
        point_ss = i #Get the step that reaches eq. point

Pop.model.viewVaccination(point_ss = point_ss, vaccination_cost=cost, filename='vaccination2')
