from hrp import HRP
from simulation import Simulation


def main():
    experiment_1 = Simulation()

    r, cols, dependency = experiment_1.generate_data(10000, 5, 5, .25)

    portfolio_1 = HRP(r)

    x = portfolio_1.weights

    print('portfolio Weights: \n')
    print(x)


if __name__ == '__main__':
    main()
