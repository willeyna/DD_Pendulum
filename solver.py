import numpy as np
import matplotlib.pyplot as plt
from math import *


class Solver():

    def __init__(self, r0, v0, Forces, m=1):
        self.r0 = np.array(r0, dtype=np.float128)
        self.v0 = np.array(v0, dtype=np.float128)

        #sets starting position and velocity solutions as unsolved for printing
        self.position = "Unsolved"
        self.velocity = "Unsolved"

        #system's forces
        self.Forces = Forces

        #object's mass
        self.mass = m;

        #dimensionality of the problem fed in
        self.dim = self.r0.shape[0]

        self.method_used = None

    def Euler(self, dt=.001, t0=0, tf=10):

        #sets up time initialization for this method
        self.t = np.arange(t0,tf,dt)
        n = self.t.shape[0]

        #init of position and velocity solutions to be filled by a method
        self.position = np.zeros([n, self.dim])
        self.velocity = np.zeros([n, self.dim])

        self.position[0,:] = self.r0
        self.velocity[0,:] = self.v0

        #loops through every time step and evaluate Euler's method
        for i in range(n-1):
            F = self.Forces(self.position, self.velocity, self.t, i, self.mass)

            self.velocity[i+1,:] = self.velocity[i,:] + dt*F/self.mass
            self.position[i+1,:] = self.position[i,:] + dt*self.velocity[i,:]
        self.method_used = "Euler's Method"


    def Verlet(self, dt=.001, t0=0, tf=10):
        #sets up time initialization for this method
        self.t = np.arange(t0,tf,dt)
        n = self.t.shape[0]

        #init of position and velocity solutions to be filled by a method
        self.position = np.zeros([n, self.dim])
        self.velocity = np.zeros([n, self.dim])

        self.position[0,:] = self.r0
        self.velocity[0,:] = self.v0

        #loops through every time step and evaluate Euler's method
        for i in range(n-1):
            F1 = self.Forces(self.position, self.velocity, self.t, i, self.mass)
            #calculates the next step in position
            self.position[i+1,:] = self.position[i,:] + dt*self.velocity[i,:] + (dt**2 / 2)*(F1/self.mass)
            #calculates the force for i+1
            F2 = self.Forces(self.position, self.velocity, self.t, i+1, self.mass)
            #uses i and i+1 acc to calculate velocity
            self.velocity[i+1,:] = self.velocity[i,:] + (dt/2)* (F1+F2)/self.mass

        self.method_used = "Velocity Verlet Method"

    def EulerCromer(self, dt=.001, t0=0, tf=10):

        #sets up time initialization for this method
        self.t = np.arange(t0,tf,dt)
        n = self.t.shape[0]

        #init of position and velocity solutions to be filled by a method
        self.position = np.zeros([n, self.dim])
        self.velocity = np.zeros([n, self.dim])

        self.position[0,:] = self.r0
        self.velocity[0,:] = self.v0

        #loops through every time step and evaluate Euler's method
        for i in range(n-1):
            F = self.Forces(self.position, self.velocity, self.t, i, self.mass)

            self.velocity[i+1,:] = self.velocity[i,:] + dt*F/self.mass
            self.position[i+1,:] = self.position[i,:] + dt*self.velocity[i+1,:]
        self.method_used = "Euler-Cromer Method"

    def RK4(self, dt=.001, t0=0, tf=10):
        #sets up time initialization for this method
        self.t = np.arange(t0,tf,dt)
        n = self.t.shape[0]

        #init of position and velocity solutions to be filled by a method
        self.position = np.zeros([n, self.dim])
        self.velocity = np.zeros([n, self.dim])

        self.position[0,:] = self.r0
        self.velocity[0,:] = self.v0

        #this is almost certaintly the wrong way to go about this but I'm in too deep
        #and too dedicated to the way I originally wrote my force functions to work in n-dim
        k1x = np.zeros_like(self.position)
        k2x = np.zeros_like(self.position)
        k3x = np.zeros_like(self.position)
        k4x = np.zeros_like(self.position)

        k1v = np.zeros_like(self.position)
        k2v = np.zeros_like(self.position)
        k3v = np.zeros_like(self.position)
        k4v = np.zeros_like(self.position)

        xx = np.zeros_like(self.position)
        vv = np.zeros_like(self.position)


        for i in range(n-1):
    # Setting up k1
            k1x[i] = self.velocity[i,:]*dt
            k1v[i] = dt*self.Forces(self.position, self.velocity, self.t, i, self.mass)
    # Setting up k2
            vv[i] = self.velocity[i,:]+k1v[i]*0.5
            xx[i] = self.position[i,:]+k1x[i]*0.5
            k2x[i] = dt*vv[i]
            k2v[i] = dt*self.Forces(xx,vv,self.t+dt*0.5, i, self.mass)
    # Setting up k3
            vv[i] = self.velocity[i,:]+k2v[i]*0.5
            xx[i] = self.position[i,:]+k2x[i]*0.5
            k3x[i] = dt*vv[i]
            k3v[i] = dt*self.Forces(xx,vv,self.t+dt*0.5, i, self.mass)
    # Setting up k4
            vv[i] = self.velocity[i,:]+k3v[i]
            xx[i] = self.position[i,:]+k3x[i]
            k4x[i] = dt*vv[i]
            k4v[i] = dt*self.Forces(xx,vv,self.t+dt, i, self.mass)
    # Final result
            self.position[i+1,:] = self.position[i,:]+(k1x[i]+2*k2x[i]+2*k3x[i]+k4x[i])/6.
            self.velocity[i+1,:] = self.velocity[i,:]+(k1v[i]+2*k2v[i]+2*k3v[i]+k4v[i])/6.
        self.method_used = "RK4 Method"

    def Plot(self, mode = "time", V = True):
        '''
        Plots the motion of the solved system in a variety of ways.
        input "mode":
        "time": Plots the positions and velocities in each direction with respect to time
        "2d": Plots the x-y positions against each other
        "3d": Plots the x-y-z position in 3d
        "phase": Plots the position magnitude against velocity magnitude in phase space
        '''
        if mode == "time":
            for i in range(self.dim):
                fig = plt.figure(figsize = (15,5))
                plt.xlabel("Time")
                plt.ylabel("Position")
                plt.title(f"Position vs Time for Dimension {i}")
                plt.plot(self.t, self.position[:,i])
                plt.show()
                if V:
                    fig = plt.figure(figsize = (15,5))
                    plt.xlabel("Time")
                    plt.ylabel("Velocity")
                    plt.title(f"Velocity vs Time for Dimension {i}")
                    plt.plot(self.t, self.velocity[:,i])
                    plt.show()

        elif mode == "2d":
            if self.dim >= 2:
                fig = plt.figure(figsize = (15,5))
                plt.xlabel("X Position")
                plt.ylabel("Y Position")
                plt.title("2D Position Over Time")
                plt.axis('equal')
                plt.plot(self.position[:,0],self.position[:,1])
            else:
                print("ERROR: Dimensionality of the problem is too small.")

        elif mode == '3d':
            if self.dim >= 3:
                fig = plt.axes(projection='3d')
                fig.set_xlabel('x')
                fig.set_ylabel('y')
                fig.set_zlabel('z')
                # Graph the flight path, add label and legend
                fig.scatter(self.position[:,0],self.position[:,1],self.position[:,2])
            else:
                print("ERROR: Dimensionality of the problem is too small.")

        elif mode == "phase":
            for i in range(self.dim):
                fig = plt.figure(figsize = (10,10))
                plt.xlabel(f"Position (dim {i})")
                plt.ylabel(f"Velocity (dim {i})")
                plt.title(f"Phase Space Plot for Dimension {i}")
                plt.plot(self.position[:,i], self.velocity[:,i])
                plt.show()
        else:
            print("Unrecognized plotting mode.")


    def __repr__(self):
        try:
            return f"""
This system is modeling a {self.dim}-dimensional problem with forces detailed in the external Forces() function.\n
This object started with position: {self.r0} and velocity: {self.v0}.\n
The final position of this solved system is {self.position[-1,:]} with velocity {self.velocity[-1,:]}.\n
This system was solved using {self.method_used}.
            """
        except:
            return f"""
This system is modeling a {self.dim}-dimensional problem with forces detailed in the external Forces() function.\n
This object started with position: {self.r0} and velocity: {self.v0}.\n
This system has not yet been solved with any method.
            """
