from spinup.utils import plot

plot.make_plots(all_logdirs=["./logs"],
                xaxis="step",
                values=["AverageTestEpRet"])