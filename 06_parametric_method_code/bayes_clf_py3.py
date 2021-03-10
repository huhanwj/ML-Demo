# -*- coding: utf-8 -*-
__author__ = "CSCI3320TA"
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.mlab as mlab
from numpy.random import normal
import seaborn as sns
sns.set()



def plot_distribution(handler, x_, y1_, y2_, saveplot_):

    fig = plt.figure(handler)
    ax = plt.gca()
    plt.plot(x_,
             y1_,
             sns.xkcd_rgb["windows blue"],
             label="Class $C_1$",
             linewidth=3)
    plt.plot(x_,
             y2_,
             sns.xkcd_rgb["pale red"],
             label="Class $C_2$",
             linewidth=3)
    legend = plt.legend(frameon=1, loc='best', fontsize=20)
    frame = legend.get_frame()
    frame.set_color('white')
    plt.title('Theoretical Distribution of Two Classes', fontsize=20)
    plt.xlabel(r'$x$', fontsize=20)
    plt.ylabel(r'$p(x|C_1)$ and $p(x|C_2)$', fontsize=20)
    plt.grid(True)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    if saveplot_:
        plt.savefig('dist_t.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_posterior(handler, x_, y1_, y2_, saveplot_):

    fig = plt.figure(handler)
    ax = plt.gca()
    plt.plot(x_,
             y1_,
             sns.xkcd_rgb["windows blue"],
             label="Class $C_1$",
             linewidth=3)
    plt.plot(x_,
             y2_,
             sns.xkcd_rgb["pale red"],
             label="Class $C_2$",
             linewidth=3)
    legend = plt.legend(frameon=1, loc='best', fontsize=20)
    frame = legend.get_frame()
    frame.set_color('white')
    plt.title('Theoretical Posterior Probability of Two Classes', fontsize=20)
    plt.xlabel(r'$x$', fontsize=20)
    plt.ylabel(r'$p(C_1|x)$ and $p(C_2|x)$', fontsize=20)
    plt.grid(True)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    if saveplot_:
        plt.savefig('post_t.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_points_and_dist(handler, x_, y1_, y2_, s1_, s2_, saveplot_):

    fig = plt.figure(handler)
    ax = plt.gca()
    plt.plot(x_,
             y1_,
             sns.xkcd_rgb["windows blue"],
             label="Class $C_1$",
             linewidth=3)
    plt.plot(x_,
             y2_,
             sns.xkcd_rgb["pale red"],
             label="Class $C_2$",
             linewidth=3)
    plt.scatter(s1_,
                np.zeros(len(s1_)),
                marker='D',
                color=sns.xkcd_rgb["windows blue"],
                s=40,
                label="Data Sample in $C_1$")
    plt.scatter(s2_,
                np.zeros(len(s2_)),
                marker='o',
                color=sns.xkcd_rgb["pale red"],
                s=40,
                label="Data Sample in $C_2$")
    legend = plt.legend(frameon=1, loc='best', fontsize=20)
    frame = legend.get_frame()
    frame.set_color('white')
    plt.title('Theoretical Posterior Probability of Two Classes', fontsize=20)
    plt.xlabel(r'$x$', fontsize=20)
    plt.ylabel(r'$p(C_1|x)$ and $p(C_2|x)$', fontsize=20)
    plt.grid(True)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.xlim([0, 6])
    plt.ylim([-0.05, 1.5])
    if saveplot_:
        plt.savefig('datasample_t.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_post_and_est_dist(handler, x_, y1_, y2_, ye1_, ye2_, saveplot_):

    fig = plt.figure(handler)
    ax = plt.gca()
    l1, = plt.plot(x_,
                   y1_,
                   sns.xkcd_rgb["windows blue"],
                   label="$p(C_1|x)$",
                   linewidth=3)
    l2, = plt.plot(x_,
                   y2_,
                   sns.xkcd_rgb["pale red"],
                   label="$p(C_2|x)$",
                   linewidth=3)
    l3, = plt.plot(x_,
                   ye1_,
                   linestyle='--',
                   color=sns.xkcd_rgb["windows blue"],
                   label="$\hat p(C_1|x)$",
                   linewidth=3)
    l4, = plt.plot(x_,
                   ye2_,
                   linestyle='--',
                   color=sns.xkcd_rgb["pale red"],
                   label="$\hat p(C_2|x)$",
                   linewidth=3)
    legend = plt.legend(frameon=1,
                        handles=[l1, l2, l3, l4],
                        loc=1,
                        fontsize=20)
    frame = legend.get_frame()
    frame.set_color('white')
    plt.title('Theoretical Posterior Probability of Two Classes', fontsize=20)
    plt.xlabel(r'$x$', fontsize=20)
    plt.ylabel(r'$p(C_1|x)$ and $p(C_2|x)$', fontsize=20)
    plt.grid(True)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.xlim([0, 4])
    plt.ylim([-0.05, 1.3])
    if saveplot_:
        plt.savefig('compare_post.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_thdist_and_est_dist(handler, x_, y1_, y2_, ye1_, ye2_, s1_, s2_,
                             saveplot_):

    fig = plt.figure(handler)
    ax = plt.gca()
    l1, = plt.plot(x_,
                   y1_,
                   sns.xkcd_rgb["windows blue"],
                   label="$p(C_1|x)$",
                   linewidth=3)
    l2, = plt.plot(x_,
                   y2_,
                   sns.xkcd_rgb["pale red"],
                   label="$p(C_2|x)$",
                   linewidth=3)
    l3, = plt.plot(x_,
                   ye1_,
                   linestyle='--',
                   color=sns.xkcd_rgb["windows blue"],
                   label="$\hat p(C_1|x)$",
                   linewidth=3)
    l4, = plt.plot(x_,
                   ye2_,
                   linestyle='--',
                   color=sns.xkcd_rgb["pale red"],
                   label="$\hat p(C_2|x)$",
                   linewidth=3)
    plt.scatter(s1_,
                np.zeros(len(s1_)),
                marker='D',
                color=sns.xkcd_rgb["windows blue"],
                s=40,
                label="Data Sample in $C_1$")
    plt.scatter(s2_,
                np.zeros(len(s2_)),
                marker='o',
                color=sns.xkcd_rgb["pale red"],
                s=40,
                label="Data Sample in $C_2$")
    legend = plt.legend(frameon=1,
                        handles=[l1, l2, l3, l4],
                        loc='best',
                        fontsize=20)
    frame = legend.get_frame()
    frame.set_color('white')
    plt.title('Theoretical Posterior Probability of Two Classes', fontsize=20)
    plt.xlabel(r'$x$', fontsize=20)
    plt.ylabel(r'$p(C_1|x)$ and $p(C_2|x)$', fontsize=20)
    plt.grid(True)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.xlim([0, 6])
    plt.ylim([-0.05, 1.5])
    if saveplot_:
        plt.savefig('compare_t.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def intersection_point(mu1_, sigma1_, mu2_, sigma2_):
    '''
    Calculate the intersection points using the quadratic formula.
    '''
    a = 1.0 / (2.0 * sigma1_ ** 2) - 1.0 / (2.0 * sigma2_ ** 2)
    b = mu2_ / (sigma2_ ** 2) - mu1_ / (sigma1_ ** 2)
    c = (0.5 * mu1_ ** 2) / (sigma1_ ** 2) - (0.5 * mu2_ ** 2) / (
        sigma2_ ** 2) + math.log(
            sigma1_ / sigma2_)
    d = math.sqrt(b ** 2 - 4 * a * c)

    intersect1 = (-b + d) / (2 * a)
    intersect2 = (-b - d) / (2 * a)

    return intersect1, intersect2


def main():
    # set the random state
    np.random.seed(2016)
    # initial parameters (unknown to the classifier)
    mu1, sigma1 = 3.5, 1.0
    mu2, sigma2 = 1.5, 0.3

    # Generate theoreticalal normal distribution output
    x = np.arange(0, 6, 0.05)
    y1_t, y2_t = mlab.normpdf(x, mu1, sigma1), mlab.normpdf(x, mu2, sigma2)
    # print(x, y1_t, y2_t)

    # Theoreticalal posteriors
    post_y1_t, post_y2_t = y1_t / (y1_t + y2_t), y2_t / (y1_t + y2_t)

    # Generate samples of normal distribution random numbers
    N = 15
    sample1, sample2 = normal(mu1, sigma1, N), normal(mu2, sigma2, N)

    # Parameter estimations
    mu1_hat, sigma1_hat = np.mean(sample1), np.std(sample1)
    mu2_hat, sigma2_hat = np.mean(sample2), np.std(sample2)

    # Get normal distribution output with estimated parameters
    y1_hat, y2_hat = mlab.normpdf(x, mu1_hat, sigma1_hat), mlab.normpdf(
        x, mu2_hat, sigma2_hat)

    # Estimated posteriors
    post_y1_hat, post_y2_hat = y1_hat / (y1_hat + y2_hat), y2_hat / (
        y1_hat + y2_hat)
    # Theoretical intersection points
    x1_t, x2_t = intersection_point(mu1, sigma1, mu2, sigma2)
    # Estimated intersection points
    x1_hat, x2_hat = intersection_point(mu1_hat, sigma1_hat, mu2_hat,
                                        sigma2_hat)
    # Get the visualization
    saveplot = True
    plot_distribution(1, x, y1_t, y2_t, saveplot)
    plot_posterior(2, x, post_y1_t, post_y2_t, saveplot)
    plot_post_and_est_dist(3, x, post_y1_t, post_y2_t, post_y1_hat, post_y2_hat, saveplot)
    plot_points_and_dist(4, x, y1_t, y2_t, sample1, sample2, saveplot)
    plot_thdist_and_est_dist(5, x, y1_t, y2_t, y1_hat, y2_hat, sample1, sample2, saveplot)

    # Print values to the std output
    print("True values mu1 = {0}, sigma1 = {1}, mu2 = {2}, sigma2 = {3}".format(
        mu1, sigma1, mu2, sigma2))
    print("Estimated values mu1 = {0}, sigma1 = {1}, mu2 = {2}, sigma2 = {3}".format(
        mu1_hat, sigma1_hat, mu2_hat, sigma2_hat))
    print("Theoretical intersection points: x1 = {0}, x2 = {1}".format(x1_t, x2_t))
    print("Estimcated intersection points: x1 = {0}, x2 = {1}".format(x1_hat, x2_hat))


if __name__ == '__main__':
    main()
