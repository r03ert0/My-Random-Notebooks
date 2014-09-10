import numpy as np
import spd_manifold

def log_lik(subject_cov, group_cov, sigma, whiten=True):
    if whiten:
        whitening = spd_manifold.inv_sqrtm(group_cov)
        subject_cov = np.dot(np.dot(whitening, subject_cov), whitening)
        group_cov = np.eye(n_rois)
    return ( -n_rois**2*np.log(sigma)
              - 1/(2*sigma**2) * (
                    np.sum((group_cov - subject_cov)**2)
                  + np.sum(np.diag(group_cov - subject_cov)**2)
            ))


class CovRFX(object):

    def __init__(self, whiten=True):
        self.whiten = whiten

    def fit(self, group_covs):
        if self.whiten:
            self.mean_cov = mean_cov = spd_manifold.log_mean(group_covs)
            whitening = spd_manifold.inv_sqrtm(mean_cov)
            group_covs = [np.dot(np.dot(whitening, g), whitening)
                        for g in group_covs]
            mean_cov = np.eye(n_rois)
        else:
            self.mean_cov = mean_cov = np.mean(group_covs, axis=0)
        self.sigma = 1./n_rois* np.sqrt(1./len(group_covs)*
                            np.sum(
                                np.sum((mean_cov - g)**2)
                                + np.sum(np.diag(mean_cov - g)**2)
                                for g in group_covs
                         ))
        return self

    def log_lik(self, subject_cov):
        return log_lik(subject_cov, self.mean_cov, self.sigma,
                        whiten=self.whiten)


if __name__ == '__main__':
    WHITEN = True

    # load the controls
    control_covs = np.load('controls.npy')
    control_covs = np.mean(control_covs, 1)
    n_controls, n_rois, _ = control_covs.shape

    # load the patients
    patient_covs = np.load('patients.npy')
    patient_covs = np.mean(patient_covs, 1)
    n_patients = len(patient_covs)
    patient_nbs = [4, 13, 18, 15, 16, 20, 22, 27, 30, 36]

    # 'test on control and patients'
    control_model = CovRFX(whiten=WHITEN).fit(control_covs)

    stop
    control_fits = [control_model.log_lik(c) for c in control_covs]
    patient_fits = [control_model.log_lik(p) for p in patient_covs]

    patient_fit_cv = np.zeros(n_patients)
    control_fit_cv = list()
    
    for n in range(n_controls):
        train = [control_covs[i] 
                 for i in range(n_controls) if i!=n]
        test = control_covs[n]
        control_model.fit(train)
        control_fit_cv.append(control_model.log_lik(test))
        patient_fit_cv += np.array([control_model.log_lik(p) 
                                    for p in patient_covs])

    patient_fit_cv /= n_controls
    
    import matplotlib.pylab as pl
    pl.rcParams['text.usetex'] = True
    pl.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
    pl.figure(1, figsize=(1, 3))
    pl.clf()
    ax = pl.axes([.2, .2, .5, .7])
    pl.boxplot([control_fit_cv, patient_fit_cv], widths=.25)
    pl.plot(1.26*np.ones(len(control_fit_cv)), control_fit_cv, '+k',
            markeredgewidth=1)
    pl.plot(2.26*np.ones(len(patient_fits)),
            patient_fit_cv, '+k',
            markeredgewidth=1)
    pl.xticks((1.13, 2.13), ('controls', 'patients'), size=13)
    if WHITEN:
        title = 'Tangent\nspace'
    else:
        title = r'$\mathbb{R}^{n\times n}$'
    pl.text(.1, .1, title,
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='bottom',
            size=12)
    #pl.axis([0.7, 2.5, 401, 799])
    pl.xlim(.7, 2.5)
    #pl.ylim(401, 799)
    ax.yaxis.tick_right()
    pl.ylabel('Log-likelihood', size=13)
    ax.yaxis.set_label_position('right')
    pl.draw()
    #ax.yaxis.set_ticks_position('both')
    pl.show()
    pl.draw()
    if WHITEN:
        pl.savefig('model_likelihood_tangent.pdf') 
    else:
        pl.savefig('model_likelihood_flat.pdf') 

