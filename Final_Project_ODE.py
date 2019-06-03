# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

'''
rho = 28.0
sigma =10.0
beta = 8.0 / 3.0

def f(state, t):
  x, y, z = state  # unpack the state vector
  return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # derivatives

state0 = [5, 5, 5]
t = np.arange(0.0, 40.0, 0.01)

states = odeint(f, state0, t)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(states[:,0], states[:,1], states[:,2])
plt.show()
'''
#=============================1=======================================
#                     Solving ODE
#=====================================================================
#!/bin/python2.7
"""

solve second order, homogeneous ODE: damped, harmonic oscillator

- compare analytical and numerical (both Euler and Range Kutta)

ODE: my''(t) + ky(t) = 0
        f(t) = F*cos(wt)
        w0**2 = k/m


"""

#--------------fct definitions--------------------------
def runge_kutta_vec( tn, Yn, fct_RHS, params):
    """
    fourth order runge kutta stepper, for single or system of  ODEs
    :input       tn           - current time step
                 Yn           - function value at time tn
                 fct_RHS      - vectorised function that defines slope (derivative) at point (t, y)
                                = all RHS of the system of ODEs

                 params   - py dictionary that includes step size 'h'
                            and all parameters needed in function:
                            fct_RHS

    :return: a_fn1 - values of derivatives at current tn for all ODEs
    """
    h = params['h']
    Kn1 = fct_RHS( tn, Yn,   params)
    Kn2 = fct_RHS( tn + .5*h, Yn + .5*h*Kn1, params)
    Kn3 = fct_RHS( tn + .5*h, Yn + .5*h*Kn1, params)
    Kn4 = fct_RHS( tn + h   , Yn + h*Kn1   , params)
    return (Kn1 + 2*Kn2 + 2*Kn3 + Kn4)/6

def lorenz( t, Yn, par):
    """
    - describe second order ODE for forced, undamped oscillation by two first order ODEs
        ODE:  y''(t) + w0**2y(t) = f(t)
                                f(t) = F*cos(w*t)
             u   =  y; v = y'
             u'  =  v
             v'  = -k*u + f(t)
    :input - t - time vector
             u  - displacement
             v  - velocity
             F  - amplitude of forcing fct
             w0 -  parameter: natural fequency: w0**2 = k/m

    :return:  [displ, vel]
    """
    x, y, z = Yn[0], Yn[1], Yn[2]
    #fn1 = v
    #fn2 = -par['w0']**2*u - par['gamma']*v
    dx_dt = par['sigma']* (-x + y)
    dy_dt = (par['rho']*x) - y- (x*z)
    dz_dt = (-par['beta']*z) + (x*y)
    return np.array([dx_dt, dy_dt, dz_dt])

def num_sol( at, y0, par):
    """
    - solve second order ODE for forced, undamped oscillation by solving two first order ODEs
       ODE:  y''(t) + ky(t) = f(t)
                              f(t) = F*cos(w*t)
    :param y0:         - IC
    :param at  :       - time vector
    :param par :       - dictionary with fct parameters
    :return: ay_hat    - forward Euler
             ay_hat_rk = 4th order runge kutta
    """
    nSteps    = at.shape[0]
    # create vectors for displacement and velocity
    a_x = np.zeros( nSteps)
    a_y = np.zeros( nSteps)
    a_z = np.zeros( nSteps)
    # set initial conditions
    a_x[0] = y0[0]
    a_y[0] = y0[1]
    a_z[0] = y0[2]
    for i in range( nSteps-1):
        # slope at previous time step, i
        fn1, fn2, fn3 = runge_kutta_vec( at[i], np.array([a_x[i], a_y[i], a_z[i]]), lorenz, dPar)
        # forward Euler: y[n+1] = y[n] + fn*h
        #fn1 = av_hat[i]
        #fn2 = -par['w0']**2*au_hat[i]

        # Integration step: Runge Kutta or Euler formula
        a_x[i+1] = a_x[i] + fn1*dPar['h']
        a_y[i+1] = a_y[i] + fn2*dPar['h']
        a_z[i+1] = a_z[i] + fn3*dPar['h']
    return a_x, a_y, a_z


#-------------------------------1----------------------------------------
#                   params, files, dir
#------------------------------------------------------------------------
dPar = { #frequencies
        'sigma' : 10,
        'rho'   : 28,
        'beta'  : 8./3.,
        # initial conditions for displ. and velocity
        'y01' : 1, 'y02' : 1, 'y03' : 1,
        # time stepping
        'h'      : 1e-2,
        'tStart' : 0,
        'tStop'  : 20*np.pi}

#--------------------------------2---------------------------------------
#                      analytical solution
#------------------------------------------------------------------------
a_t = np.arange( dPar['tStart'], dPar['tStop']+dPar['h'], dPar['h'])
#ay_ana    = dPar['y01']*np.cos(dPar['w0']*a_t)
#a_ana = (-dPar['rho']*dPar['y01'])+ (dPar['rho'] + 1)*dPar['y02']

#--------------------------------3---------------------------------------
#                      numerical solutions
#------------------------------------------------------------------------

ax_num, ay_num, az_num = num_sol(  a_t, [dPar['y01'], dPar['y02'], dPar['y03']], dPar)
#--------------------------------4---------------------------------------
#                            plots
#------------------------------------------------------------------------
plt.figure(1)
ax1 = plt.axes(projection='3d')
ax1.plot3D(ax_num, ay_num, az_num, c='r')
'''
plt.figure(2)
ax = plt.subplot( 111) #plt.axes( [.12, .12, .83, .83])
ax.plot( a_t, ax_num,   'k-', lw = 3, alpha = .3, label = 'num - displ.')
#ax.plot( a_t,  a_ana, 'r--', lw = 1, label = 'ana')
ax.set_xlabel( 'Time [s]')
ax.set_ylabel( 'Displacement [mm]')
ax.legend( loc = 'upper left')
plt.show()
'''

