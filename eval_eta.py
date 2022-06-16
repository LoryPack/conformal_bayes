import numpy as np

examples = ['sparsereg_diabetes', 'sparsereg_diabetes_misspec', ]
# examples = ['sparsereg_diabetes_misspec', ]
etas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

method = 'cb'

# Report Conformal results#
for example in examples:

    print('EXAMPLE: {}'.format(example))
    for eta in etas:
        suffix = method + '_' + example
        if eta != 1:
            suffix += "_eta_" + str(eta)

        # Coverage
        coverage = np.mean(np.load("results/coverage_{}.npy".format(suffix)), axis=1)  # take mean over test values
        rep = np.shape(coverage)[0]
        mean = np.mean(coverage)
        se = np.std(coverage) / np.sqrt(rep)
        print("{} coverage is {:.3f} ({:.3f})".format(eta, mean, se))

        # Return exact coverage if cb
        if method == 'cb' and (example not in ['sparseclass_breast', 'sparseclass_parkinsons']):
            suffix_ex = method + '_exact_' + example
            if eta != 1:
                suffix_ex += "_eta_" + str(eta)
            coverage = np.mean(np.load("results/coverage_{}.npy".format(suffix_ex)),
                               axis=1)  # take mean over test values
            rep = np.shape(coverage)[0]
            mean = np.mean(coverage)
            se = np.std(coverage) / np.sqrt(rep)
            print("{} exact coverage is {:.3f} ({:.3f})".format(eta, mean, se))
    print()

    for eta in etas:
        suffix = method + '_' + example
        if eta != 1:
            suffix += "_eta_" + str(eta)
        # Length
        length = np.mean(np.load("results/length_{}.npy".format(suffix)), axis=1)
        rep = np.shape(length)[0]
        mean = np.mean(length)
        se = np.std(length) / np.sqrt(rep)
        print("{} length is {:.2f} ({:.2f})".format(eta, mean, se))  # Times
    print()

    # for eta in etas:
    #     suffix = method + '_' + example
    #     if eta != 1:
    #         suffix += "_eta_" + str(eta)
    #     # Length
    #     times = np.load("results/times_{}.npy".format(suffix))
    #     rep = np.shape(times)[0]
    #     mean = np.mean(times)
    #     se = np.std(times) / np.sqrt(rep)
    #     print("{} times is {:.3f} ({:.3f})".format(eta, mean, se))  # Times
    # print()
