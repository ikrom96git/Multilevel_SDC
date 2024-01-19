import numpy as np
import matplotlib.pyplot as plt
from core.Pars import _Pars
class HarmonicOscillator:
    def __init__(self, problem_params):
        self.params=_Pars(problem_params)
        self.determinant=self.params.mu**2*0.25-self.params.kappa
    def get_righthandside(self):
        pass

    def get_exact_freeOscillation(self, time:float)->np.ndarray:
        
        if self.params.mu==0 and self.params.kappa>0.0:
            omega=np.sqrt(self.params.kappa)
            pos_withoutFriction=a0*np.cos(omega*time)+b0*np.sin(omega*time)
            vel_withoutFriction=-omega*a0*np.sin(omega*time)+omega*b0*np.cos(omega*time)
            return [pos_withoutFriction, vel_withoutFriction]

        elif self.params.mu>0 and self.params.kappa>0:
            Determinant=self.params.mu**2*0.25-self.params.kappa
            exponent=np.exp(self.params.mu*0.5*time)
            if Determinant<0:
                omegaTilde=np.sqrt(-Determinant)
                pos_withFriction=exponent*(a0*np.cos(omegaTilde*time)+b0*np.sin(omegaTilde*time))
                vel_withFriction=exponent*(a0*np.cos(omegaTilde*time)+b0*np.sin(omegaTilde*time))
                
            elif Determinant>0:
                omegaTilde=np.sqrt(Determinant)
                pos_withFriction=exponent*(a0*np.cos(omegaTilde*time)+b0*np.sin(omegaTilde*time))
                vel_withFriction=exponent*(a0*np.cos(omegaTilde*time)+b0*np.sin(omegaTilde*time))

            elif Determinant==0:
                pos_withFriction=exponent*(a0+b0*time)
                vel_withFriction=exponent*(a0+b0*time)
            return [pos_withFriction, vel_withFriction]
        else:
            reise ValueError("The parameters are not valid for a harmonic oscillator.")
  
    def get_determinantPositive(self):
        omegaTilde=np.sqrt(-self.determinant)
        

    
