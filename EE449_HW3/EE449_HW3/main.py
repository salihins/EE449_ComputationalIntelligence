import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import skfuzzy.control as ctrl
from vaccination import Vaccination
#Create universal variables to use in Antecedent and Consequent functions

#vacc_per = np.linspace(0, 1.0, 0.05)
#sigma = np.linspace(-0.2, 0.2, 0.05)
#linspace caused the following error.
#TypeError: 'float' object cannot be interpreted as an integer
#Hence, it is replaced with np.arange which is taken from:
#https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem_newapi.html
class myFuzzy():
    def __init__(self):
        self.total = None
        self.model = Vaccination()
        self.getPercent()
        self.getCost()
        #Create the two fuzzy variables one input and one output
        self.current_percentage = ctrl.Antecedent(np.arange(0,1.01,0.05),'current_percentage')
        self.out_sig = ctrl.Consequent(np.arange(-0.2,0.21,0.05),'out_sig')

        self.current_percentage['low'] = fuzz.trimf(self.current_percentage.universe, [0, 0.4, 0.6])
        self.current_percentage['average'] = fuzz.trimf(self.current_percentage.universe, [0.4, 0.6, 0.8])
        self.current_percentage['high'] = fuzz.trimf(self.current_percentage.universe, [0.6, 0.8, 1])
        self.current_percentage.view()

        self.out_sig['low'] = fuzz.trimf(self.out_sig.universe, [-0.2, -0.2, 0])
        self.out_sig['average'] = fuzz.trimf(self.out_sig.universe, [-0.2, 0, 0.2])
        self.out_sig['high'] = fuzz.trimf(self.out_sig.universe, [0, 0.2, 0.2])
        self.out_sig.view()
        #Rule1: If the vaccination rates are high, then we do not need to highly control the rates.
        #Hence, if the vaccination rate is high, output is low.

        # Rule2: If the vaccination rates are low, then we need to highly control the rates.
        # Hence, if the vaccination rate is low, output is high.

        # Rule3: If the vaccination rates are average, then we need to regularly control the rates.
        # Hence, if the vaccination rate is average, output is average.

        self.rule1 = ctrl.Rule(antecedent = self.current_percentage['high'], consequent=self.out_sig['low'])
        self.rule2 = ctrl.Rule(antecedent =self.current_percentage['low'], consequent=self.out_sig['high'])
        self.rule3 = ctrl.Rule(antecedent =self.current_percentage['average'], consequent=self.out_sig['average'])

        self.vacc_ctrl = ctrl.ControlSystem([self.rule1, self.rule2, self.rule3])


    def FuzzyLogic(self):
        self.fuzz = ctrl.ControlSystemSimulation(self.vacc_ctrl)
        self.percentage = self.model.checkVaccinationStatus()[0]
        self.fuzz.input['vacc_rates'] = self.percentage
        self.fuzz.compute()
        self.outputControl = self.fuzz.output['control']
        self.model.vaccinatePeople(self.outputControl)
        """
        self.current_percentage['low'] = fuzz.interp_membership(self.current_percentage, self.current_percentage['low'], self.percentage)
        self.current_percentage['high'] = fuzz.interp_membership(self.current_percentage, self.current_percentage['high'], self.percentage)
        self.current_percentage['average'] = fuzz.interp_membership(self.current_percentage,  self.current_percentage['average'], self.percentage)

        self.rule1 = np.fmax(self.current_percentage['high'], self.out_sig['low'])
        self.rule2 = np.fmax(self.current_percentage['low'], self.out_sig['high'])
        self.rule3 = np.fmax( self.current_percentage['average'], self.out_sig['average'])

        self.total = np.fmax(self.rule1, np.fmax(self.rule2, self.rule3))
     

        
        self.output_control = fuzz.defuzz(self.current_percentage, self.total, 'centroid')
        self.model.vaccinatePeople(self.output_control)
        """
    def getPercent(self):
        self.percentage = self.model.checkVaccinationStatus()[0]
        return self.percentage
    def getCost(self):
        self.cost = self.percentage = self.model.checkVaccinationStatus()[0]
        return self.cost

Pop = myFuzzy()
# Take the necessa
flag = True  # Flag will be True until equilibrium point
cost = 0
stopDiff = 0.0005
for i in range(0,201):
    Pop.FuzzyLogic()  # Apply the control by calling the method
    Pop.getPercent()
    Pop.getCost()# Update variables so that we can check vacc percentage and calculate costs
    if (flag == True):
        cost += Pop.cost  # Sum all costs until equilibrium point
    currentDiff = abs(Pop.percentage - 0.6)  # Calculate the diff between percentage and 60
    if (currentDiff < stopDiff) and flag == True:  # Check if the difference is enough small
        flag = False  # Flag will be Flase when equilibrium point is reached
        point_ss = i  # record the time when there is a equibilirium

Pop.model.viewVaccination(point_ss= point_ss, vaccination_cost=cost, filename='vaccination1')