import numpy as np
import logging
logger = logging.getLogger(__name__)

class Eigenproblem():
    def __init__(self, EVP, reject=True, factor=1.5, scales=1, drift_threshold=1e6, use_ordinal=False, grow_func=lambda x: x.real, freq_func=lambda x: x.imag):
        """An object for feature-rich eigenvalue analysis.

        Eigenproblem provides support for common tasks in eigenvalue
        analysis. Dedalus EVP objects compute raw eigenvalues and
        eigenvectors for a given problem; Eigenproblem provides support for
        numerous common tasks required for scientific use of those
        solutions. This includes rejection of inaccurate eigenvalues and
        analysis of those rejection criteria, plotting of eigenmodes and
        spectra, and projection of 1-D eigenvectors onto 2- or 3-D domains
        for use as initial conditions in subsequent initial value problems.

        Additionally, Eigenproblems can compute epsilon-pseudospectra for
        arbitrary Dedalus differential-algebraic equations.


        Parameters
        ----------
        EVP : dedalus.core.problems.EigenvalueProblem
            The Dedalus EVP object containing the equations to be solved
        reject : bool, optional
            whether or not to reject spurious eigenvalues (default: True)
        factor : float, optional
            The factor by which to multiply the resolution.
            NB: this must be a rational number such that factor times the
            resolution of EVP is an integer. (default: 1.5)
        scales : float, optional
            A multiple for setting the grid resolution.  (default: 1)
        drift_threshold : float, optional
            Inverse drift ratio threshold for keeping eigenvalues during
            rejection (default: 1e6)
        use_ordinal : bool, optional
            If true, use ordinal method from Boyd (1989); otherwise use
            nearest (default: False)
        grow_func : func
            A function that takes a complex input and returns the growth
            rate as defined by the EVP (default: uses real part)
        freq_func : func
            A function that takes a complex input and returns the frequency
            as defined by the EVP (default: uses imaginary part)

        Attributes
        ----------
        evalues : ndarray
            Lists "good" eigenvalues
        evalues_low : ndarray
            Lists eigenvalues from low resolution solver (i.e. the
            resolution of the specified EVP)
        evalues_high : ndarray
            Lists eigenvalues from high resolution solver (i.e. factor
            times specified EVP resolution)
        pseudospectrum : ndarray
            epsilon-pseudospectrum computed at specified points in the
            complex plane
        ps_real : ndarray
            real coordinates for epsilon-pseudospectrum
        ps_imag : ndarray
            imaginary coordinates for epsilon-pseudospectrum

        Notes
        -----
        See references for algorithms in individual method docstrings.

        """
        self.reject = reject
        self.factor = factor
        self.EVP = EVP
        #self.solver = EVP.build_solver()
        #if self.reject:
        #    self._build_hires()

        #self.grid_name = self.EVP.domain.bases[0].name
        self.evalues = None
        self.evalues_low = None
        self.evalues_high = None
        self.pseudospectrum = None
        self.ps_real = None
        self.ps_imag = None

        self.drift_threshold = drift_threshold
        self.use_ordinal = use_ordinal
        self.scales = scales
        self.grow_func = grow_func
        self.freq_func = freq_func

    def discard_spurious_eigenvalues_via_tau(self):
        """ Use an experimental tau-thresholding to reject spurious eigenvalues.
        Returns trustworthy eigenvalues using the L2 magnitude of the taus.
        """
        evals = self.evalues
        taus = self.taus

        # Reverse engineer correct indices to make unsorted list from sorted
        indices = np.arange(len(evals))

        evals_indices_taus = np.asarray(list(zip(evals, indices, taus)))

        # remove NaNs:
        evals_indices_taus = evals_indices_taus[np.isfinite(evals_indices_taus[:,0])]

        # sort by real part of eigenvalue:
        evals_indices_taus = evals_indices_taus[np.argsort(evals_indices_taus[:,0].real)]

        # reject via tau amplitudes
        mask_ok = np.where(evals_indices_taus[:,-1] < self.tau_cutoff) # np.where()?
        evals_indices_taus_ok = evals_indices_taus[mask_ok]
        evals_ok = evals_indices_taus_ok[:, 0]
        indices_ok = evals_indices_taus_ok[:, 1].real.astype(int)
        logger.debug(f'mode reject: {evals_ok[-1]:.3g}, {indices_ok[-1]}, rejecting {evals.size-evals_ok.size}, retained {evals_ok.size} good eigenvalues with |τ|_2 < {self.tau_cutoff:.3g}')

        self.tau_amplitudes_sorted = evals_indices_taus[:,-1]
        return evals_ok, indices_ok

    def plot_taus(self, axes=None):
        """Plot tau amplitudes vs. mode number.

        The L2 tau amplitudes give a measure of how well a given eigenmode is
        resolved; this is experimental, buy may be able to help set thresholds.

        Returns
        -------
        matplotlib.figure.Figure

        """
        if self.reject is False:
            raise NotImplementedError("Can't plot tau amplitudes unless eigenvalue rejection is True.")

        if axes is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axes
            fig = axes.figure

        taus = self.tau_amplitudes_sorted
        mode_numbers = np.arange(len(taus))
        ax.semilogy(mode_numbers, taus,'o',alpha=0.4)
        good_taus = np.where(taus < self.tau_cutoff)
        ax.semilogy(mode_numbers[good_taus], taus[good_taus],'o', label='tau')
        ax.axhline(self.tau_cutoff,alpha=0.4, color='black')
        ax.set_xlabel("mode number")
        ax.set_ylabel(r"$|\tau|_2$")
        ax.legend()

        return ax


    def discard_spurious_eigenvalues(self):
        """ Solves the linear eigenvalue problem for two different
        resolutions.  Returns trustworthy eigenvalues using nearest delta,
        from Boyd chapter 7.
        """
        eval_low = self.evalues_low
        eval_hi = self.evalues_high

        # Reverse engineer correct indices to make unsorted list from sorted
        reverse_eval_low_indx = np.arange(len(eval_low))
        reverse_eval_hi_indx = np.arange(len(eval_hi))

        eval_low_and_indx = np.asarray(list(zip(eval_low, reverse_eval_low_indx)))
        eval_hi_and_indx = np.asarray(list(zip(eval_hi, reverse_eval_hi_indx)))

        # sort taus
        taus = np.array(self.taus)
        taus = taus[np.isfinite(eval_low)]

        # remove nans
        eval_low_and_indx = eval_low_and_indx[np.isfinite(eval_low)]
        eval_hi_and_indx = eval_hi_and_indx[np.isfinite(eval_hi)]

        # sort taus on NaN-cleaned
        taus = taus[np.argsort(eval_low_and_indx[:, 0].real)]
        self.tau_amplitudes_sorted = taus

        # Sort eval_low and eval_hi by real parts
        eval_low_and_indx = eval_low_and_indx[np.argsort(eval_low_and_indx[:, 0].real)]
        eval_hi_and_indx = eval_hi_and_indx[np.argsort(eval_hi_and_indx[:, 0].real)]

        eval_low_sorted = eval_low_and_indx[:, 0]
        eval_hi_sorted = eval_hi_and_indx[:, 0]

        # Compute sigmas from lower resolution run (gridnum = N1)
        sigmas = np.zeros(len(eval_low_sorted))
        sigmas[0] = np.abs(eval_low_sorted[0] - eval_low_sorted[1])
        sigmas[1:-1] = [0.5*(np.abs(eval_low_sorted[j] - eval_low_sorted[j - 1]) + np.abs(eval_low_sorted[j + 1] - eval_low_sorted[j])) for j in range(1, len(eval_low_sorted) - 1)]
        sigmas[-1] = np.abs(eval_low_sorted[-2] - eval_low_sorted[-1])

        if not (np.isfinite(sigmas)).all():
            logger.warning("At least one eigenvalue spacings (sigmas) is non-finite (np.inf or np.nan)!")

        # Ordinal delta
        self.delta_ordinal = np.array([np.abs(eval_low_sorted[j] - eval_hi_sorted[j])/sigmas[j] for j in range(len(eval_low_sorted))])

        # Nearest delta
        self.delta_near = np.array([np.nanmin(np.abs(eval_low_sorted[j] - eval_hi_sorted)/sigmas[j]) for j in range(len(eval_low_sorted))])

        # Discard eigenvalues with 1/delta_near < drift_threshold
        if self.use_ordinal:
            inverse_drift = 1/self.delta_ordinal
        else:
            inverse_drift = 1/self.delta_near
        eval_low_and_indx = eval_low_and_indx[np.where(inverse_drift > self.drift_threshold)]
        drift_ratios = inverse_drift[np.where(inverse_drift > self.drift_threshold)]

        eval_low = eval_low_and_indx[:, 0]
        indx = eval_low_and_indx[:, 1].real.astype(int)
        logger.debug(f'mode reject: {eval_low[-1]:.3g}, {indx[-1]}, {drift_ratios[-1]:3g}, rejecting {eval_low_sorted.size-eval_low.size} modes, keeping {eval_low.size}')
        return eval_low, indx

    def plot_drift_ratios(self, axes=None):
        """Plot drift ratios (both ordinal and nearest) vs. mode number.

        The drift ratios give a measure of how good a given eigenmode is;
        this can help set thresholds.

        Returns
        -------
        matplotlib.figure.Figure

        """
        if self.reject is False:
            raise NotImplementedError("Can't plot drift ratios unless eigenvalue rejection is True.")

        if axes is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axes
            fig = axes.figure

        mode_numbers = np.arange(len(self.delta_near))
        ax.semilogy(mode_numbers,1/self.delta_near,'o',alpha=0.4)
        ax.semilogy(mode_numbers,1/self.delta_ordinal,'x',alpha=0.4)

        ax.set_prop_cycle(None)
        good_near = 1/self.delta_near > self.drift_threshold
        good_ordinal = 1/self.delta_ordinal > self.drift_threshold
        ax.semilogy(mode_numbers[good_near],1/self.delta_near[good_near],'o', label='nearest')
        ax.semilogy(mode_numbers[good_ordinal],1/self.delta_ordinal[good_ordinal],'x',label='ordinal')
        ax.axhline(self.drift_threshold,alpha=0.4, color='black')
        ax.set_xlabel("mode number")
        ax.set_ylabel(r"$1/\delta$")
        ax.legend()

        return ax

    def plot_drift_ratios_vs_taus(self, axes=None):
        """Plot drift ratios (both ordinal and nearest) vs. tau amplitudes.

        The drift ratios give a measure of how good a given eigenmode is;
        this can help set thresholds.

        Returns
        -------
        matplotlib.figure.Figure

        """
        if self.reject is False:
            raise NotImplementedError("Can't plot drift ratios unless eigenvalue rejection is True.")

        if axes is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axes
            fig = axes.figure

        taus = self.tau_amplitudes_sorted
        mode_numbers = np.arange(len(self.delta_near))
        ax.loglog(taus,1/self.delta_near,'o',alpha=0.4)
        ax.loglog(taus,1/self.delta_ordinal,'x',alpha=0.4)

        ax.set_prop_cycle(None)
        good_near = 1/self.delta_near > self.drift_threshold
        good_ordinal = 1/self.delta_ordinal > self.drift_threshold
        ax.loglog(taus[good_near],1/self.delta_near[good_near],'o', label='nearest')
        ax.loglog(taus[good_ordinal],1/self.delta_ordinal[good_ordinal],'x',label='ordinal')
        ax.axhline(self.drift_threshold,alpha=0.4, color='black')
        ax.axvline(self.tau_cutoff,alpha=0.4, color='black')

        ax.set_xlabel(r"$|\tau|_2$")
        ax.set_ylabel(r"$1/\delta$")
        ax.legend()

        return ax
