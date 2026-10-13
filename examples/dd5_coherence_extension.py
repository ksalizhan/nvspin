"""Compare PDD, CPMG, and XY8 under correlated frequency noise."""

from nvspin.control import (
    cpmg,
    pdd,
    xy8,
)

from nvspin.experiments import (
    DynamicalDecoupling,
)

from nvspin.noise import (
    OrnsteinUhlenbeckNoise,
)


def main():

    noise = OrnsteinUhlenbeckNoise(
        sigma=0.002,             # 2 MHz
        correlation_time=1000.0, # 1 us
    )

    sequences = [
        pdd(8),
        cpmg(8),
        xy8(1),
    ]

    print()
    print("DD coherence under OU noise")
    print("---------------------------")

    for sequence in sequences:

        experiment = DynamicalDecoupling(
            sequence=sequence,
            noise=noise,
            tau=100.0,
            realizations=300,
            dt=5.0,
            seed=12345,
        )

        result = experiment.simulate()

        print(
            f"{sequence.name:10s} "
            f"T = {result.total_time:7.1f} ns   "
            f"L = {result.coherence:.5f}"
        )


if __name__ == "__main__":
    main()