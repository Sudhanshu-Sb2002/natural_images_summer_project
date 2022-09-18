import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit

c = 0.0256 / 10 ** (-12)
a = (1 * 20 + 0.04 * 440) / 1000
b = (1 * 20 + 0.04 * 50) / 1000
p_x = 0.5
y = 500 / 1000
x_0 = 50 / 1000


@njit
def func(t, x):
    inside = (a + p_x * y) / (b + p_x * x) * x / y
    return -c * np.log(inside)

@njit
def range_kutta(func, xvals, tvals):
    t = tvals[0]
    x = xvals[0]
    h = tvals[1] - tvals[0]
    for i in range(len(tvals) - 1):
        k1 = h * func(t, x)
        k2 = h * func(t + h / 2, x + k1 / 2)
        k3 = h * func(t + h / 2, x + k2 / 2)
        k4 = h * func(t + h, x + k3)
        x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = tvals[i + 1]
        xvals[i + 1] = x
    return xvals

def V(x):
    return c * np.log((a + p_x * y) / y * np.divide(x, (b + p_x * x)))


def main():
    # initial conditions
    t_min=0
    t_max=0.01
    n_times=10000
    tvals = np.linspace(t_min, t_max, n_times)
    xvals = np.zeros(n_times)
    xvals[0] = x_0
    xvals = range_kutta(func, xvals, tvals)
    plt.plot(tvals, xvals)
    plt.show()

    plt. plot(xvals, V(xvals))
    plt.show()

    plt.plot(tvals, V(xvals))
    plt.show()

def not_main(text):
    for i in range(len(text)):
        if text[i]==':':
            for j in range(i+1, len(text)):
                if text[j]=='\n':
                    print(text[i+1:j])
                    break

if __name__ == "__main__":
    text=''''•	Materials Research Centre
    o	Faculty
    	N. Ravishankar: nravi@mrc.iisc.ernet.in 
    	Karuna Kar Nanda: nanda@mrc.iisc.ernet.in 
    	Bikramjit Basu: bikram@mrc.iisc.ernet.in 
    	Sujit Das: sujitdas@iisc.ac.in 
    	Rajamalli P.: rajamalli@iisc.ac.in 
    	Subinoy Rana: subinoy@iisc.ac.in 
    	Prabeer Barpanda: prabeer@mrc.iisc.ac.in 
    	Balram Sahoo: bsahoo@mrc.iisc.ernet.in 
    	Abhishek Singh: abhishek@iisc.ac.in 
    o	INSA Faculty
    	S. B. Krupanidhi: sbk@mrc.iisc.ernet.in 
    o	Associate Faculty
    	C. N. R. Rao: Could not find IISc Email
    o	Retired Faculty
    	K. B. R. Varma: kbrvarma@mrc.iisc.ernet.in 
    	G. Ananthakrishna: garani@mrc.iisc.ernet.in
    	Arun M. Umarji:  umarji@mrc.iisc.ernet.in
    •	Materials Engineering
    o	Faculty
    	Satyam Suwas: satyamsuwas@iisc.ac.in 
    	Abhik Choudhury: abhiknc@iisc.ac.in 
    	Abinandanan T. A.: abinand@iisc.ac.in 
    	Aloke Paul: aloke@iisc.ac.in 
    	Ankur Chauhan: ankurchauhan@iisc.ac.in 
    	Ashok M. Raichur: amr@iisc.ac.in 
    	Avadhani G. S.: gsa@iisc.ac.in 
    	Bhagwati Prasad: bpjoshi@iisc.ac.in 
    	Chandan Srivastava: csrivastava@iisc.ac.in 
    	Chokshi A. H.: achokshi@iisc.ac.in 
    	Deshpande R. J.: rjd@iisc.ac.in 
    	Govind S. Gupta: govindg@iisc.ac.in 
    	Karthikeyan S.: karthik@iisc.ac.in 
    	Kaushik Chatterjee: kchatterjee@iisc.ac.in 
    	Padaikathan P.: padai@iisc.ac.in 
    	Praveen C. Ramamurthy: praveen@iisc.ac.in 
    	Praveen Kumar: praveenk@iisc.ac.in 
    	Prosenjit Das: prosenjitdas@iisc.ac.in 
    	Rajeev Ranjan: rajeev@iisc.ac.in 
    	Ravi R.: rravi@iisc.ac.in
    	Sachin R. Rondiya: rondiya@iisc.ac.in 
    	Sai Gautam Gopalakrishnan: saigautamg@iisc.ac.in 
    	Subho Dasgupta: dasgupta@iisc.ac.in 
    	Subodh Kumar: skumar@iisc.ac.in 
    	Surendra Kumar M.: skmakineni@iisc.ac.in 
    	Suryasarathi Bose: sbose@iisc.ac.in 
    o	Visiting and Adjunct Faculty
    	Sonal Asthana: sonalasthana@iisc.ac.in 
    	Shervanthi Homer-Vanniasinkam
    	Prasad K. D. V. Yarlagadda
    	R. Gopalan
    o	Honorary Faculty
    	Dipankar Banerjee: dbanerjee@iisc.ac.in 
    	Vikram Jayaram: qjayaram@iisc.ac.in
    o	Inspire Faculty
    	Upashi Goswami: upashig@iisc.ac.in 
    o	Emeritus Faculty
    	Kamanio Chattopadhyay: kamanio@iisc.ac.in 
    	Jacob K. T.: katob@iisc.ac.in 
    	Natarajan K. A.: kan@iisc.ac.in 
    	Ranganathan S.: rangu@iisc.ac.in 
    	Subramanian S.: ssmani@iisc.ac.in 
    •	Instrumentation and Applied Physics
    o	Faculty
    	S. Asokan: sasokan@iisc.ac.in
    	Partha Pratim Mondal: partha@iisc.ac.in 
    	Sanjiv Sambandan: sanjiv@iisc.ac.in 
    	Sai Siva Gorthi: saisiva@iisc.ac.in 
    	Jayaprakash: jayap@iisc.ac.in
    •	Centre for Nano Science and Engineering
    o	Faculty
    	Aditya Sadhanala: sadhanala@iisc.ac.in
    	Chandan Kumar: kchandan@iisc.ac.in
    	Supradeepa V. R: supradeepa@iisc.ac.in
    •	Electrical Communications Engineering
    o	Faculty
    	T. Srinivas: tsrinu@iisc.ac.in
    	Varun Raghunathan: varunr@iisc.ac.in
    o	Principal Research Scientist
    	E. S. Shivaleela: lila@iisc.ac.in
    '''
    not_main(text)
