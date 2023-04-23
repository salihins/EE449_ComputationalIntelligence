import numpy as np
import skfuzzy as fuzz
from vaccination import Vaccination
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

#Class declaration
class myFuzzy():
    def __init__(self):
        #Call vaccination.py
        self.model = Vaccination()
        #Update variable
        self.UpdatePercent()
        self.UpdateCost()
        # Update variable

        self.vacc_set = np.arange(0, 1.01, 0.01)
        self.cont_rate = np.arange(-0.2, 0.21, 0.05)
        #https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html#example-plot-tipping-problem-py
        #above website is used and some parts are directly taken
        self.low_vacc = fuzz.trapmf(self.vacc_set, [0, 0, 0.4, 0.6])
        self.avg_vacc = fuzz.trimf(self.vacc_set, [0.4, 0.6, 0.8])
        self.high_vacc = fuzz.trapmf(self.vacc_set, [0.6, 0.8, 1, 1])
        #https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html#example-plot-tipping-problem-py
        #above website is used and some parts are directly taken
        self.high_cont = fuzz.trapmf(self.cont_rate, [0, 0.1, 0.2, 0.2])
        self.avg_cont = fuzz.trimf(self.cont_rate, [-0.1, 0, 0.1])
        self.low_cont = fuzz.trapmf(self.cont_rate, [-0.2, -0.2, -0.1, 0])

    def FuzzyLogic(self):
        #https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html#example-plot-tipping-problem-py
        #Directly taken from here. I just adjusted the variable names.
        self.low_level = fuzz.interp_membership(self.vacc_set, self.low_vacc, self.vacc_perc)
        self.avg_level = fuzz.interp_membership(self.vacc_set, self.avg_vacc, self.vacc_perc)
        self.high_level = fuzz.interp_membership(self.vacc_set, self.high_vacc, self.vacc_perc)


        #Overall rule is that the vaccination rate and control rate is inversely proportional.
        #As one of them increases, decrease the other one.

        #Rule1: If vaccination is low, then we need the have high control.
        self.rule1 = np.fmin(self.low_level, self.high_cont)
        #Rule2: If vaccination is average, then we can decrease the control to average.
        self.rule2 = np.fmin(self.avg_level, self.avg_cont)
        #Rule3: If vaccination is high, then we can decrease the control to low.
        self.rule3 = np.fmin(self.high_level, self.low_cont)
        # https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html#example-plot-tipping-problem-py
        # Directly taken from here. I just adjusted the variable names.


        #Directly taken from: https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html#example-plot-tipping-problem-py
        self.aggregated = np.fmax(self.rule1, np.fmax(self.rule2, self.rule3))
        self.output_control = fuzz.defuzz(self.cont_rate, self.aggregated, 'centroid')

        self.model.vaccinatePeople(self.output_control)
    def UpdatePercent(self):
        #Update variable
        self.vacc_perc = self.model.checkVaccinationStatus()[0]
    def UpdateCost(self):
        # Update variable
        self.cost = self.model.vaccination_rate_curve_[-1]

#Create a variable for check to eq. point.
checker = 1
#Initialize (declare) cost
cost = 0
#Define error value
error = 0.0001
#Call the fuzzy system
Pop = myFuzzy()
for i in range(200):
    # Initialize logic
    Pop.FuzzyLogic()
    #Update variable
    Pop.UpdatePercent()
    Pop.UpdateCost()
    # Update variable
    if checker == 1:
        cost += Pop.cost #Update the cost until eq. point
    diff = abs(Pop.vacc_perc - 0.6)  #Check the difference
    if (diff < error) and (checker == 1):#If eq. conditions are satisfied
        checker = 0
        checker = 0
        point_ss = i #Get the step that reaches eq. point

Pop.model.viewVaccination(point_ss = point_ss, vaccination_cost=cost, filename='vaccination1')
