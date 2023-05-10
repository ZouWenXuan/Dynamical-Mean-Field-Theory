# Dynamical-Mean-Field-Theory

Numerical solution for dynamical mean-field equation of generic random neural networks. The dynamics of original system reads,
$$
\begin{equation}
	\dot{x}_i(t)=-x_i(t)+g \sum_{j=1} J_{i j} \phi_j(t) + j_i(t) +\sigma \xi_i(t), \quad i=1, \ldots, N,
\end{equation}
$$
where the Gaussian noise $\xi_i(t)$ has the variance $\left\langle \xi_i(t)\xi_j(t^{\prime})\right\rangle =\delta_{ij}\delta(t-t^{\prime})$. Connections $J_{ij}$ are drawn from the centered Gaussian distribution with the variance $\frac{1}{N}$ as well as the covariance $\overline{J_{ij} J_{ji}}  = \frac{\eta}{N}$.  The mean-field dynamics reads,
$$
\begin{equation}
	\dot{x}(t) = -x(t) + \gamma(t) +  g^2 \eta  \int_{0}^{t} R(t,s)\phi(s)    \mathrm {d}s,
\end{equation}
$$
where $\gamma(t)$ is the effective noise with the temporally correlated variance,
$$
\begin{equation}
	\langle\gamma(t)\gamma(t^{\prime})\rangle = g^2 C(t,t^{\prime}) + \sigma^2\delta(t-t^{\prime}).
\end{equation}
$$


# Requirements

Python 3.8.5



# Some Instructions

- The model folder covers the code for numerical solution of DMFT as well as RNN simulation.  
- Type "python main.py" to get a comparison between numerical solution and simulated results.
- Please contact me if you have any questions about this code. My email: zouwx5@mail2.sysu.edu.cn



# Citation

This code is the product of work carried out by the group of [PMI lab, Sun Yat-sen University](https://www.labxing.com/hphuang2018). If the code helps, consider giving us a shout-out in your publications.