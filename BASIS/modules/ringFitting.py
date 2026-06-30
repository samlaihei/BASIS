import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from uncertainties import ufloat
from scipy import integrate
import GCFit.modules.imutils as imutils

class RingFit():
    def __init__(self, imgarr, fov=225, interp_factor=2, blur_uas=0, total_flux=1):
        """Class to fit ring features in an image array.

        Parameters
        ----------
        imgarr : np.ndarray or eh.image
            2D image array to fit ring features on.
        fov : float, optional
            Field of view of the image in microarcseconds, by default 225.
        interp_factor : int, optional
            Factor by which to interpolate the image for fitting, by default 2.
        blur_uas : float, optional
            Amount of Gaussian blur to apply to the image in microarcseconds, by default 0.
        total_flux : float, optional
            Total flux to normalize the image to, by default 1.
        """
        self.imgarr = imgarr
        self.interp_factor = interp_factor
        self.pol_img = None
        self.center = None
        self.mean_rpk = None
        self.best_metric = None
        self.ring_width = None
        self.ring_orientation = None
        self.azimuthal_asymmetry = None
        self.frac_central_brightness = None

        if isinstance(imgarr, np.ndarray):
            self.imgdim = imgarr.shape[0]
            self.fov = fov
            self.regrid_img = imutils.gauss_blur(imgarr, sigma=blur_uas/2.355, psize=self.fov/self.imgdim)
            self.regrid_img -= np.min(self.regrid_img)
            self.regrid_img = self.regrid_img / np.sum(self.regrid_img) * total_flux
            temp_centroid = imutils.centroid(self.regrid_img)
            self.regrid_img = imutils.shift_fft(self.regrid_img, shift=(self.imgdim/2 - temp_centroid[0], self.imgdim/2 - temp_centroid[1]))

        else:
            self.imgdim = self.imgarr.xdim
            self.fov = self.imgarr.fovx()*1e6*206265
            self.regrid_img = self.imgarr.regrid_image(self.fov/206265/1e6, self.imgdim, interp='bicubic')
            self.regrid_img = self.regrid_img.blur_circ(blur_uas*1e-6/206265)
            self.regrid_img._imdict['I'] -= np.min(self.regrid_img._imdict['I'])
            self.regrid_img._imdict['I'] = self.regrid_img._imdict['I'] / np.sum(self.regrid_img._imdict['I']) * total_flux
            self.regrid_img.center()
            self.regrid_img = self.regrid_img.imarr()

        self.imgarr = cv2.resize(self.regrid_img, (self.imgdim*self.interp_factor, self.imgdim*self.interp_factor), interpolation=cv2.INTER_CUBIC)

    def circular_mean(self, rads):
        """Compute the circular mean and standard deviation of an array of angles in radians.
        
        Parameters
        ----------
        rads : np.ndarray
            Array of angles in radians.

        Returns
        -------
        mean_angle : float
            Circular mean of the angles in radians.
        angular_stddev : float
            Circular standard deviation of the angles in radians.
        """
        mean_angle = np.angle(np.mean(np.exp(1j * rads)))
        angular_diffs = np.angle(np.exp(1j * (rads - mean_angle)))
        angular_stddev = np.sqrt(np.mean(angular_diffs**2))
        return mean_angle, angular_stddev
    
    def convert_to_uas(self, pix): # on interp image
        """Convert pixel units to microarcseconds on the interpolated image.
        
        Parameters
        ----------
        pix : float
            Pixel value on the interpolated image.

        Returns
        -------
        float
            Value in microarcseconds.
        """
        return pix / (self.imgdim * self.interp_factor) * self.fov

    def convert_to_pix(self, uas): # on interp image
        """Convert microarcseconds to pixel units on the interpolated image.
        
        Parameters
        ----------
        uas : float
            Value in microarcseconds.

        Returns
        -------
        float
            Pixel value on the interpolated image.
        """
        return uas / self.fov * (self.imgdim * self.interp_factor)

    def make_polar(self, imgarr, center):
        """Convert an image to polar coordinates.

        Parameters
        ----------
        imgarr : np.ndarray
            2D image array to convert.
        center : tuple
            (x, y) coordinates of the center for polar conversion.

        Returns
        -------
        np.ndarray
            Polar coordinate image array.
        """
        pol_img = cv2.linearPolar(imgarr, center, imgarr.shape[-1]//2, cv2.WARP_FILL_OUTLIERS)
        pol_img -= np.min(pol_img)
        return pol_img
    
    def compute_metric(self, imgarr, center):
        """Compute the ring fitting metric for a given center.
        
        Parameters
        ----------
        imgarr : np.ndarray
            2D image array.
        center : tuple
            (x, y) coordinates of the center.

        Returns
        -------
        float
            Ring fitting metric.
        float
            Mean radius of peak brightness in microarcseconds.
        float
            Standard deviation of radius of peak brightness in microarcseconds.
        """
        pol_img = self.make_polar(imgarr, center)
        mean_rpk, std_rpk, metric = self.rpk(pol_img)
        mean_rpk_uas = mean_rpk/self.imgdim/self.interp_factor * self.fov/2
        std_rpk_uas = std_rpk/self.imgdim/self.interp_factor * self.fov/2
        return metric, mean_rpk_uas, std_rpk_uas

    def rpk(self, pol_img):
        """Compute the radius of peak brightness (RPK) metric.
        
        Parameters
        pol_img : np.ndarray
            Polar coordinate image array.
        
        Returns
        -------
        float
            Mean radius of peak brightness in pixels.
        float
            Standard deviation of radius of peak brightness in pixels.
        float
            RPK metric (std/mean).
        """
        rpks = np.argmax(pol_img, axis=1)
        mean_rpk, std_rpk = np.mean(rpks), np.std(rpks)
        metric = std_rpk/mean_rpk
        return mean_rpk, std_rpk, metric

    def compute_center(self, center_constraint=20, verbose=False):
        """Compute the best-fit center of the ring feature in the image.

        Parameters
        ----------
        center_constraint : float, optional
            Radius in microarcseconds to constrain the search for the center, by default 20.
        verbose : bool, optional
            Whether to print the best-fit center and metrics, by default False.
        
        Returns
        -------
        np.ndarray
            Best-fit (x, y) center coordinates.
        ufloat
            Mean radius of peak brightness with uncertainty in microarcseconds.
        float
            Best RPK metric.
        """
        xcoords = np.arange(0, self.imgarr.shape[1], step=1, dtype=float)
        ycoords = np.arange(0, self.imgarr.shape[0], step=1, dtype=float)
        best_metric = np.inf
        best_center = None
        best_mean_rpk = None
        best_std_rpk = None

        coords = [(x, y) for x in xcoords for y in ycoords]
        # constrain to central 20 uas region
        center_region_pix = self.convert_to_pix(center_constraint)*2
        center_coords = []
        for coord in coords:
            if np.sqrt((coord[0]-self.imgarr.shape[1]/2)**2 + (coord[1]-self.imgarr.shape[0]/2)**2) < center_region_pix:
                center_coords.append(coord)

        coords = center_coords

        for coord in tqdm(coords, disable=not verbose, desc='Computing center'):
            metric, mean_rpk_uas, std_rpk_uas = self.compute_metric(self.imgarr, coord)
            if metric < best_metric:
                best_metric = metric
                best_center = coord
                best_mean_rpk = mean_rpk_uas
                best_std_rpk = std_rpk_uas
        best_center = np.array([best_center[0]/self.interp_factor-1/self.interp_factor, best_center[1]/self.interp_factor-1/self.interp_factor])
        self.center = best_center
        self.mean_rpk = ufloat(best_mean_rpk, best_std_rpk)
        self.best_metric = best_metric
        if verbose:
            print(f'\nBest Center: {self.center}, Mean RPK: {self.mean_rpk:.2f} uas, Metric: {self.best_metric:.4f}')
        return self.center, self.mean_rpk, self.best_metric

    def compute_ring_width(self, background_uas=None, verbose=False):
        """Compute the ring width of the ring feature in the image.
        
        Parameters
        ----------
        background_uas : float, optional
            Radius in microarcseconds to estimate background intensity, by default 50.
        verbose : bool, optional
            Whether to print the ring width, by default False.

        Returns
        -------
        ufloat
            Ring width with uncertainty in microarcseconds.
        """
        if background_uas is not None:
            r_background_pix = self.convert_to_pix(background_uas)*2
        elif self.mean_rpk is not None:
            r_background_pix = self.convert_to_pix(2*self.mean_rpk.n)*2
        else:
            r_background_pix = self.convert_to_pix(50)*2

        center_pix = [(self.center[0]+1/self.interp_factor)*self.interp_factor, (self.center[1]+1/self.interp_factor)*self.interp_factor]
        pol_img = self.make_polar(self.imgarr, center_pix)
        background_intensity = np.mean(pol_img[int(r_background_pix)-1:int(r_background_pix)+1, :])
        sub_pol_img = pol_img - background_intensity
        sub_pol_img[sub_pol_img < 0] = 0
        def fwhm(radial_profile):
            half_max = np.max(radial_profile)/2
            above_half_max = np.where(radial_profile >= half_max)[0]
            if len(above_half_max) < 2:
                return 0
            fwhm_val = above_half_max[-1] - above_half_max[0]
            return fwhm_val
        fwhm_values = []
        for azimuthal_idx in range(sub_pol_img.shape[0]):
            radial_profile = sub_pol_img[azimuthal_idx, :]
            fwhm_val = fwhm(radial_profile)
            fwhm_values.append(fwhm_val)
        mean_fwhm = self.convert_to_uas(np.mean(fwhm_values))/2
        std_fwhm = self.convert_to_uas(np.std(fwhm_values))/2
        self.ring_width = ufloat(mean_fwhm, std_fwhm)
        if verbose:
            print(f'Ring Width: {self.ring_width:.2f} uas')
        return self.ring_width

    def compute_ring_orientation(self, verbose=False):
        """Compute the ring orientation of the ring feature in the image.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print the ring orientation, by default False.

        Returns
        -------
        ufloat
            Ring orientation with uncertainty in radians.
        """
        mean_rpk_pix = self.convert_to_pix(self.mean_rpk.n)*2
        ringwidth_pix = self.convert_to_pix(self.ring_width.n)*2
        r_in = mean_rpk_pix - ringwidth_pix/2
        r_out = mean_rpk_pix + ringwidth_pix/2
        pol_img = cv2.linearPolar(self.imgarr, (self.center[0]*self.interp_factor+1, self.center[1]*self.interp_factor+1), self.imgarr.shape[-1]//2, cv2.WARP_FILL_OUTLIERS)
        pol_img -= np.min(pol_img)
        orientations = []
        for r_idx in range(int(r_in), int(r_out)+1):
            theta_slice = pol_img[:, r_idx]
            theta_vals = np.linspace(0, 2 * np.pi, theta_slice.shape[0], endpoint=False)
            integrand_real = theta_slice * np.cos(theta_vals)
            integrand_imag = theta_slice * np.sin(theta_vals)
            real_part = integrate.trapezoid(integrand_real, theta_vals)
            imag_part = integrate.trapezoid(integrand_imag, theta_vals)
            orientation = np.angle(real_part + 1j * imag_part)
            orientations.append(orientation + 2*np.pi if orientation < 0 else orientation)
        avg_orientation, std_orientation = self.circular_mean(np.array(orientations))
        self.ring_orientation = (2*np.pi - (ufloat(avg_orientation, std_orientation) + np.pi/2)) % (2*np.pi)
        if verbose:
            print(f'Ring Orientation: {np.degrees(self.ring_orientation.n):.2f} +/- {np.degrees(self.ring_orientation.s):.2f} degrees CCW from North')
        return self.ring_orientation
        
    def compute_azimuthal_asymmetry(self, verbose=False):
        """Compute the azimuthal asymmetry of the ring feature in the image.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print the azimuthal asymmetry, by default False.

        Returns
        -------
        ufloat
            Azimuthal asymmetry.
        """
        mean_rpk_pix = self.convert_to_pix(self.mean_rpk.n)*2
        ringwidth_pix = self.convert_to_pix(self.ring_width.n)*2
        r_in = mean_rpk_pix - ringwidth_pix/2
        r_out = mean_rpk_pix + ringwidth_pix/2
        pol_img = self.make_polar(self.imgarr, (self.center[0]*self.interp_factor+1, self.center[1]*self.interp_factor+1))
        azi_asyms = []
        for r_idx in range(int(r_in), int(r_out)+1):
            theta_slice = pol_img[:, r_idx]
            theta_vals = np.linspace(0, 2 * np.pi, theta_slice.shape[0], endpoint=False)
            integrand_real = theta_slice * np.cos(theta_vals)
            integrand_imag = theta_slice * np.sin(theta_vals)
            real_part = integrate.trapezoid(integrand_real, theta_vals)
            imag_part = integrate.trapezoid(integrand_imag, theta_vals)
            num = np.abs(real_part + 1j * imag_part)
            denom = integrate.trapezoid(theta_slice, theta_vals)
            azi_asyms.append(num/denom)
        mean_azi_asym = np.mean(azi_asyms)
        std_azi_asym = np.std(azi_asyms)
        self.azimuthal_asymmetry = ufloat(mean_azi_asym, std_azi_asym)
        if verbose:
            print(f'Azimuthal Asymmetry: {self.azimuthal_asymmetry:.4f}')
        return self.azimuthal_asymmetry

    def compute_frac_central_brightness(self, inner_ring_uas=None, verbose=False):
        """Compute the fractional central brightness of the ring feature in the image.

        Parameters
        ----------
        inner_ring_uas : int, optional
            Inner ring radius in microarcseconds, by default 5.
        verbose : bool, optional
            Whether to print the fractional central brightness, by default False.

        Returns
        -------
        float
            Fractional central brightness.
        """
        mean_rpk_pix = self.convert_to_pix(self.mean_rpk.n)*2
        pol_img = self.make_polar(self.imgarr, (self.center[0]*self.interp_factor+1, self.center[1]*self.interp_factor+1))
        if inner_ring_uas is not None:
            max_inner_radius = self.convert_to_pix(inner_ring_uas)*2
        elif self.mean_rpk is not None:
            max_inner_radius = self.convert_to_pix(self.mean_rpk.n/5)*2 
        else:
            max_inner_radius = self.convert_to_pix(5)*2  # 5 uas
        mean_inner_intensity = np.mean(pol_img[:, :int(max_inner_radius)])
        mean_ring_intensity = np.mean(pol_img[:, int(mean_rpk_pix)-1:int(mean_rpk_pix)+1])
        frac_brightness = mean_inner_intensity / mean_ring_intensity
        self.frac_central_brightness = frac_brightness
        if verbose:
            print(f'Fractional Central Brightness: {self.frac_central_brightness:.2e}')
        return self.frac_central_brightness

    def run_all(self, center_constraint=20, inner_ring_uas=None, background_uas=None, verbose=True):
        """Run all ring fitting computations.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print all computed parameters, by default True.
        """
        self.compute_center(center_constraint=center_constraint, verbose=verbose)
        self.compute_ring_width(background_uas=background_uas, verbose=verbose)
        self.compute_ring_orientation(verbose=verbose)
        self.compute_azimuthal_asymmetry(verbose=verbose)
        self.compute_frac_central_brightness(inner_ring_uas=inner_ring_uas, verbose=verbose)
        self.pol_img = cv2.linearPolar(self.imgarr, (self.center[0]*self.interp_factor+1, self.center[1]*self.interp_factor+1), self.imgarr.shape[-1], cv2.WARP_FILL_OUTLIERS)
        return

    def get_results(self):
        """Get the computed ring fitting results.

        Returns
        -------
        dict
            Dictionary containing the computed ring fitting parameters.
        """
        results = {
            'center_uas': (self.center - self.imgdim/2) / self.imgdim * self.fov,
            'mean_rpk_uas': self.mean_rpk,
            'ring_width_uas': self.ring_width,
            'ring_orientation_rad': self.ring_orientation,
            'azimuthal_asymmetry': self.azimuthal_asymmetry,
            'frac_central_brightness': self.frac_central_brightness
        }
        return results

    def plot_all(self, img_ax, pol_ax, cmap='afmhot', ratio=[1,1], save_path=None):
        """Plot the original image and polar image with ring fitting results.

        Parameters
        ----------
        img_ax : matplotlib.axes.Axes
            Axes to plot the original image.
        pol_ax : matplotlib.axes.Axes
            Axes to plot the polar image.
        cmap : str, optional
            Colormap to use for plotting, by default 'afmhot'.
        ratio : list, optional
            Aspect ratio [x, y] for the polar plot, by default [1, 1].
        """
        if self.pol_img is None:
            print("Run the ring fitting first using run_all() method.")
            return
        img_ax.imshow(self.regrid_img, origin='upper', cmap=cmap, interpolation='gaussian', extent=[self.fov/2, -self.fov/2, -self.fov/2, self.fov/2])
        shifted_pol = np.flip(self.pol_img.T, axis=1)
        shifted_pol = np.roll(shifted_pol, -shifted_pol.shape[0]//4, axis=1)
        pol_ax.imshow(shifted_pol[:int(self.pol_img.shape[1]*ratio[1]/ratio[0]), :], origin='lower', cmap=cmap, interpolation='gaussian')
        img_ax.set_xlabel('Relative RA ($\\mu$as)')
        img_ax.xaxis.set_label_position('top')
        img_ax.set_ylabel('Relative Dec ($\\mu$as)')
        img_ax.tick_params(axis='both', which='both', direction='in', labeltop=True, labelbottom=False, top=True, left=True, right=True, rotation=0)
        pol_ax.tick_params(axis='both', which='both', direction='in', top=False, left=True, right=True)

        mean_rpk_pix = self.mean_rpk.n / self.fov * self.imgdim + 0.5/self.interp_factor
        center_uas = (self.center - self.imgdim/2) / self.imgdim * self.fov
        center_uas = -center_uas  # flip y axis for plotting
        center_uas -= self.fov/(self.imgdim*self.interp_factor)  # shift center for pixelization
        circle = plt.Circle(center_uas, self.mean_rpk.n, color='cyan', fill=False, linestyle='--', linewidth=2)
        img_ax.add_artist(circle)
        pol_ax.axhline(mean_rpk_pix * self.interp_factor , color='cyan', linestyle='--', linewidth=2)
        
        circ_in = plt.Circle(center_uas, (self.mean_rpk.n - self.ring_width.n/2), color='cyan', fill=False, linestyle='--', linewidth=2, alpha=0.3)
        circ_out = plt.Circle(center_uas, (self.mean_rpk.n + self.ring_width.n/2), color='cyan', fill=False, linestyle='--', linewidth=2, alpha=0.3)
        img_ax.add_artist(circ_in)
        img_ax.add_artist(circ_out)
        pol_ax.axhline((mean_rpk_pix - self.ring_width.n/2 / self.fov * self.imgdim) * self.interp_factor, color='cyan', linestyle='--', linewidth=2, alpha=0.3)
        pol_ax.axhline((mean_rpk_pix + self.ring_width.n/2 / self.fov * self.imgdim) * self.interp_factor, color='cyan', linestyle='--', linewidth=2, alpha=0.3)

        # vertical line for orientation angle
        pol_ax.axvline((self.ring_orientation.n % (2*np.pi)) / (2*np.pi) * self.pol_img.shape[0], color='magenta', linestyle='--', linewidth=2)
        # dotted line for uncertainty
        pol_ax.axvline(((self.ring_orientation.n + self.ring_orientation.s) % (2*np.pi)) / (2*np.pi) * self.pol_img.shape[0], color='magenta', linestyle=':', linewidth=2, alpha=0.7)
        pol_ax.axvline(((self.ring_orientation.n - self.ring_orientation.s) % (2*np.pi)) / (2*np.pi) * self.pol_img.shape[0], color='magenta', linestyle=':', linewidth=2, alpha=0.7)
        # radial line for orientation angle
        ring_orientation_rad = -self.ring_orientation.n + np.pi/2  # rotate by 90 degrees to get the direction of the major axis
        img_ax.plot([center_uas[0], center_uas[0] + self.mean_rpk.n * np.cos(ring_orientation_rad)], [center_uas[1], center_uas[1] + self.mean_rpk.n * np.sin(ring_orientation_rad)], color='magenta', linestyle='--', linewidth=2)
        # dotted radial
        img_ax.plot([center_uas[0], center_uas[0] + self.mean_rpk.n * np.cos(ring_orientation_rad + self.ring_orientation.s)], [center_uas[1], center_uas[1] + self.mean_rpk.n * np.sin(ring_orientation_rad + self.ring_orientation.s)], color='magenta', linestyle=':', linewidth=2, alpha=0.7)
        img_ax.plot([center_uas[0], center_uas[0] + self.mean_rpk.n * np.cos(ring_orientation_rad - self.ring_orientation.s)], [center_uas[1], center_uas[1] + self.mean_rpk.n * np.sin(ring_orientation_rad - self.ring_orientation.s)], color='magenta', linestyle=':', linewidth=2, alpha=0.7)

        pol_ax_limit = self.pol_img.shape[1]*ratio[1]/ratio[0] 
        pol_ax.set_ylim(0, pol_ax_limit)

        internal_circ = plt.Circle(center_uas, self.mean_rpk.n/5, color='cyan', fill=False, linestyle=':', linewidth=2, alpha=0.7)
        img_ax.add_artist(internal_circ)
        external_circ = plt.Circle(center_uas, self.mean_rpk.n*2, color='cyan', fill=False, linestyle=':', linewidth=2, alpha=0.7)
        img_ax.add_artist(external_circ)
        pol_ax.axhline(self.convert_to_pix(self.mean_rpk.n/5), color='cyan', linestyle=':', linewidth=2, alpha=0.7)
        pol_ax.axhline(self.convert_to_pix(self.mean_rpk.n*2), color='cyan', linestyle=':', linewidth=2, alpha=0.7)

        # label pol_ax x axis with degrees 0 to 360
        num_ticks = 5
        tick_locs = np.linspace(0, self.pol_img.shape[0], num_ticks)
        tick_labels = [f'{int(np.degrees(angle))}' for angle in np.linspace(0, 2*np.pi, num_ticks)]
        pol_ax.set_xticks(tick_locs)
        pol_ax.set_xticklabels(tick_labels)
        pol_ax.set_xlabel('Orientation Angle (deg)')

        # pol_x y axis radius in uas
        num_yticks = 5
        margin = 10
        ytick_locs = np.linspace(0, pol_ax_limit - margin, num_yticks)
        ytick_labels = [f'{int(self.convert_to_uas(radius_pix))}' for radius_pix in np.linspace(0, pol_ax_limit - margin, num_yticks)]
        pol_ax.set_yticks(ytick_locs)
        pol_ax.set_yticklabels(ytick_labels)
        pol_ax.set_ylabel('Radius ($\\mu$as)')
        
        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', transparent=False)
        return 