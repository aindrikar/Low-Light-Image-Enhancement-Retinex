import numpy as np
import cv2
import cv2.ximgproc as xip
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skopt import gp_minimize
from skopt.space import Real, Integer
import warnings
import traceback

# Set a non-interactive backend for matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

# ============================================================================
# IMPROVED CORE FUNCTIONS
# ============================================================================

def get_gaussian_blur(img, ksize=0, sigma=1):
    """Applies a 2D Gaussian filter to an image."""
    if ksize == 0:
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
    sep_k = cv2.getGaussianKernel(ksize, sigma)
    kernel = np.outer(sep_k, sep_k)
    return cv2.filter2D(img, -1, kernel)

def get_msr_reflectance(img, sigmas=[15, 80, 250], weights=[1/3, 1/3, 1/3]):
    """
    Calculates the multi-scale reflectance map using MSR.
    Args:
        img (np.array): The input image (e.g., V channel).
        sigmas (list): List of standard deviations for Gaussian filters.
        weights (list): List of weights for each scale.
    Returns:
        np.array: The combined multi-scale reflectance map.
    """
    img = img.astype(np.float64) + 1.0
    msr_reflectance = np.zeros_like(img)

    for sigma, weight in zip(sigmas, weights):
        illumination = get_gaussian_blur(img, sigma=sigma)
        reflectance = np.divide(img, illumination + 1e-8, where=illumination > 0)
        msr_reflectance += weight * reflectance
        
    return msr_reflectance

def adaptive_gamma_correction(img, alpha=0.2, beta=2.5):
    """Applies adaptive gamma correction based on mean intensity."""
    img = np.clip(img, 0, 255).astype(np.uint8)
    mean_intensity = np.mean(img) / 255.0
    gamma = alpha + beta * (1 - mean_intensity)
    gamma = np.clip(gamma, 0.5, 3.0)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)], dtype=np.uint8).reshape((256, 1))
    return cv2.LUT(img, table)

def normalize_image(img, low_percent=2, high_percent=98):
    """Normalizes the image intensity range."""
    p2, p98 = np.percentile(img, (low_percent, high_percent))
    return np.clip((img - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)

def reconstruct_image_msr(illumination, reflectance, gain_boost=1.05):
    """
    Reconstructs the final image from illumination and MSR reflectance.
    Args:
        illumination (np.array): Illumination map [0, 255]
        reflectance (np.array): MSR reflectance map [~1.0]
        gain_boost (float): A factor to boost overall brightness.
    Returns:
        np.array: The reconstructed and normalized image.
    """
    illumination_norm = illumination.astype(np.float32) / 255.0
    reflectance_norm = reflectance.astype(np.float32)
    enhanced = illumination_norm * reflectance_norm
    enhanced = np.clip(enhanced * gain_boost, 0.0, 1.0)
    return (enhanced * 255.0).astype(np.uint8)

def psnr(original, processed):
    """Computes the PSNR between two images."""
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def compute_ssim_fixed(original, processed):
    """Computes SSIM with a fixed window size for smaller images."""
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)
    win_size = 11
    h, w = original.shape[:2]
    if min(h, w) < 64:
        win_size = max(7, (min(h, w)//8)|1)
    return ssim(original, processed, data_range=255.0,
                win_size=win_size, gaussian_weights=True,
                sigma=1.5, use_sample_covariance=False)

def adaptive_denoising(image, h_param=3, h_color_param=3, template_window_size=7, search_window_size=21):
    """Applies adaptive non-local means denoising."""
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(
            image, None, h_param, h_color_param, template_window_size, search_window_size
        )
    else:
        return cv2.fastNlMeansDenoising(
            image, None, h_param, template_window_size, search_window_size
        )

# ============================================================================
# IMPROVED IFG FOR LOL DATASET
# ============================================================================

class ImprovedIFGForLOL:
    def __init__(self, alpha=0.7, beta=0.5, gamma=0.3):
        self.alpha = alpha
        self.beta = beta  
        self.gamma = gamma
        self.validate_parameters()
        
    def validate_parameters(self):
        total = self.alpha + self.beta + self.gamma
        if total > 1.5:
            scale_factor = 1.2 / total
            self.alpha *= scale_factor
            self.beta *= scale_factor
            self.gamma *= scale_factor
            
        self.alpha = np.clip(self.alpha, 0.6, 0.85)
        self.beta = np.clip(self.beta, 0.4, 0.7)
        self.gamma = np.clip(self.gamma, 0.25, 0.5)
        
    def compute_ifg_membership(self, image):
        img_norm = np.clip(image.astype(np.float32) / 255.0, 0, 1)
        mean_brightness = np.mean(img_norm)
        adaptive_alpha = self.alpha * (1.5 - mean_brightness)
        adaptive_alpha = np.clip(adaptive_alpha, 0.5, 1.0)
        
        mu_lower = np.power(img_norm, adaptive_alpha)
        mu_upper = 1.0 - np.power(1.0 - img_norm, adaptive_alpha * 0.8)
        nu = np.power(1.0 - img_norm, self.beta)
        pi = np.clip(1.0 - mu_upper - nu, 0, 1)
        
        return mu_lower, mu_upper, nu, pi
    
    def ifg_enhancement_operator(self, mu_lower, mu_upper, nu, pi):
        enhancement_strength = self.gamma
        
        enhanced_lower = mu_lower + enhancement_strength * pi * mu_lower
        enhanced_upper = mu_upper + enhancement_strength * pi * mu_upper
        
        enhanced = (enhanced_lower + enhanced_upper) / 2.0
        enhanced = np.power(enhanced, 0.9)
        
        return np.clip(enhanced, 0, 1)
    
    def apply_ifg_preprocessing(self, image):
        if len(image.shape) == 3:
            enhanced_channels = []
            for channel in range(image.shape[2]):
                ch = image[:, :, channel]
                mu_lower, mu_upper, nu, pi = self.compute_ifg_membership(ch)
                enhanced_ch = self.ifg_enhancement_operator(mu_lower, mu_upper, nu, pi)
                enhanced_channels.append((enhanced_ch * 255).astype(np.uint8))
            
            return np.stack(enhanced_channels, axis=2)
        else:
            mu_lower, mu_upper, nu, pi = self.compute_ifg_membership(image)
            enhanced = self.ifg_enhancement_operator(mu_lower, mu_upper, nu, pi)
            return (enhanced * 255).astype(np.uint8)

class AdaptiveIFGForLOL:
    def __init__(self):
        self.ifg_generator = ImprovedIFGForLOL()
        
    def analyze_lol_characteristics(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        mean_intensity = np.mean(gray) / 255.0
        std_intensity = np.std(gray) / 255.0
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        dark_ratio = np.sum(gray < 50) / gray.size
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            s_channel = hsv[:, :, 1]
            v_channel = hsv[:, :, 2]
            
            dark_mask = v_channel < 30
            if np.sum(dark_mask) > 0:
                s_in_dark = s_channel[dark_mask]
                saturation_mean_dark = np.mean(s_in_dark)
                saturation_std_dark = np.std(s_in_dark)
                saturation_max_dark = np.percentile(s_in_dark, 95)
            else:
                saturation_mean_dark = 0
                saturation_std_dark = 0
                saturation_max_dark = 0
        else:
            saturation_mean_dark = 0
            saturation_std_dark = 0
            saturation_max_dark = 0
        
        return {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity, 
            'entropy': entropy / 8.0,
            'dark_ratio': dark_ratio,
            'local_contrast': laplacian_var,
            'is_very_dark': mean_intensity < 0.15,
            'is_low_contrast': std_intensity < 0.15,
            'saturation_mean_dark': saturation_mean_dark,
            'saturation_std_dark': saturation_std_dark,
            'saturation_max_dark': saturation_max_dark,
        }
    
    def adapt_for_lol(self, characteristics):
        mean_intensity = characteristics['mean_intensity']
        dark_ratio = characteristics['dark_ratio']
        is_very_dark = characteristics['is_very_dark']
        
        if is_very_dark or dark_ratio > 0.7:
            self.ifg_generator.alpha = 0.65
            self.ifg_generator.beta = 0.45
            self.ifg_generator.gamma = 0.45
        elif mean_intensity < 0.25:
            self.ifg_generator.alpha = 0.7
            self.ifg_generator.beta = 0.5
            self.ifg_generator.gamma = 0.35
        else:
            self.ifg_generator.alpha = 0.75
            self.ifg_generator.beta = 0.55
            self.ifg_generator.gamma = 0.3
        
        if characteristics['is_low_contrast']:
            self.ifg_generator.gamma *= 1.2
        
        self.ifg_generator.validate_parameters()
        return self.ifg_generator

# ============================================================================
# SATURATION CONFIDENCE MAP (IMPROVED)
# ============================================================================

def compute_saturation_confidence_map(low_img):
    """
    Creates a pixel-wise confidence map for saturation.
    Returns values in [0, 1] where:
    - 0 = Definitely chromatic noise (suppress completely)
    - 1 = Definitely real color (preserve fully)
    """
    hsv = cv2.cvtColor(low_img, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0].astype(np.float32)
    s_channel = hsv[:, :, 1].astype(np.float32)
    v_channel = hsv[:, :, 2].astype(np.float32)
    
    confidence_map = np.ones_like(s_channel, dtype=np.float32)
    
    # Rule 1: Very dark pixels WITH HIGH saturation = suspicious
    very_dark_mask = v_channel < 15
    very_dark_high_sat = very_dark_mask & (s_channel > 20)
    confidence_map[very_dark_high_sat] *= 0.1
    
    # Rule 1b: Very dark with LOW saturation = likely real (dark gray objects)
    very_dark_low_sat = very_dark_mask & (s_channel <= 20)
    confidence_map[very_dark_low_sat] *= 0.5
    
    # Rule 2: Dark pixels (15 <= V < 30) with high saturation = suspicious
    dark_mask = (v_channel >= 15) & (v_channel < 30)
    dark_high_sat = dark_mask & (s_channel > 40)
    confidence_map[dark_high_sat] *= 0.3
    
    # Rule 3: Moderate darkness (30 <= V < 50) - more permissive
    moderate_dark = (v_channel >= 30) & (v_channel < 50)
    sat_confidence = np.clip((s_channel - 10) / 50, 0.6, 1.0)
    confidence_map[moderate_dark] = np.minimum(
        confidence_map[moderate_dark], 
        sat_confidence[moderate_dark]
    )
    
    # Rule 4: Bright pixels (V >= 50) - trust saturation fully
    bright_mask = v_channel >= 50
    confidence_map[bright_mask] = 1.0
    
    # Rule 5: Spatial consistency check
    s_blur = cv2.GaussianBlur(s_channel, (5, 5), 1.5)
    s_variance = np.abs(s_channel - s_blur)
    high_variance = s_variance > 20
    confidence_map[high_variance] *= 0.6
    
    # Smooth the confidence map
    confidence_map = cv2.GaussianBlur(confidence_map, (7, 7), 2.0)
    
    return np.clip(confidence_map, 0.0, 1.0)

# ============================================================================
# IMPROVED OPTIMIZATION FOR LOL
# ============================================================================

def improved_bayesian_optimization_for_lol(low_img, high_img, saturation_blend_map, n_iterations=25):

    def lol_objective_function(params):
        sigma_small, sigma_med, sigma_large, guided_r, guided_eps, contrast_factor, gamma_alpha = params
        
        try:
            hsv_low = cv2.cvtColor(low_img, cv2.COLOR_BGR2HSV)
            h_channel = hsv_low[:, :, 0].astype(np.float32)
            s_channel = hsv_low[:, :, 1].astype(np.float32)
            v_channel = hsv_low[:, :, 2].astype(np.float32)

            # Denoise Hue and Saturation
            h_channel = cv2.bilateralFilter(h_channel, d=5, sigmaColor=25, sigmaSpace=15)
            s_channel = cv2.bilateralFilter(s_channel, d=5, sigmaColor=25, sigmaSpace=15)
            s_channel = normalize_image(s_channel, low_percent=5, high_percent=95)

            # CRITICAL FIX: Apply blend map BEFORE MSR
            s_channel_suppressed = s_channel * saturation_blend_map

            # MSR on V and suppressed S
            msr_reflectance_v = get_msr_reflectance(v_channel, sigmas=[sigma_small, sigma_med, sigma_large])
            msr_reflectance_s = get_msr_reflectance(
                s_channel_suppressed,
                sigmas=[sigma_small/2, sigma_med/2, sigma_large/2]
            )

            # CRITICAL FIX: Scale reflectance to [0, 255] range
            msr_reflectance_s_scaled = np.clip((msr_reflectance_s - 0.5) * 255, 0, 255)

            illumination_large = get_gaussian_blur(v_channel + 1.0, sigma=sigma_large)
            illumination_large = adaptive_gamma_correction(illumination_large, alpha=gamma_alpha)

            # Mix in same scale
            s_enhanced = 0.8 * s_channel_suppressed + 0.2 * msr_reflectance_s_scaled

            # Guided filtering on reflectance
            enhanced_refl = xip.guidedFilter(
                msr_reflectance_v.astype(np.float32), 
                msr_reflectance_v.astype(np.float32), 
                radius=int(guided_r), eps=guided_eps
            )

            enhanced_refl = np.clip(enhanced_refl * contrast_factor, 0, 255)
            enhanced_v = reconstruct_image_msr(illumination_large, enhanced_refl)

            # Recombine
            hsv_low[:, :, 0] = h_channel.astype(np.uint8)
            hsv_low[:, :, 1] = s_enhanced.astype(np.uint8)
            hsv_low[:, :, 2] = enhanced_v.astype(np.uint8)
            enhanced_img = cv2.cvtColor(hsv_low, cv2.COLOR_HSV2BGR)
            
            enhanced_gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
            high_gray = cv2.cvtColor(high_img, cv2.COLOR_BGR2GRAY)
            
            ssim_score = compute_ssim_fixed(high_gray, enhanced_gray)
            psnr_score = psnr(high_img, enhanced_img)
            
            enhanced_grad = np.gradient(enhanced_gray)
            high_grad = np.gradient(high_gray)
            grad_correlation = np.corrcoef(
                np.sqrt(enhanced_grad[0]**2 + enhanced_grad[1]**2).flatten(),
                np.sqrt(high_grad[0]**2 + high_grad[1]**2).flatten()
            )[0,1]
            grad_correlation = np.nan_to_num(grad_correlation, nan=0.0)
            
            objective = (
                0.5 * ssim_score +
                0.25 * min(psnr_score / 35.0, 1.0) +
                0.25 * max(grad_correlation, 0.0)
            )
            
            return -objective
            
        except Exception as e:
            print(f"Error in LOL optimization: {e}")
            return 1.0
    
    space = [
        Real(5, 40, name='sigma_small'),
        Real(45, 120, name='sigma_med'),
        Real(150, 300, name='sigma_large'),
        Real(3, 12, name='guided_radius'),
        Real(1e-3, 1e-1, name='guided_eps'),
        Real(1.0, 1.4, name='contrast_factor'),
        Real(0.15, 0.45, name='gamma_alpha')
    ]
    
    result = gp_minimize(lol_objective_function, space, n_calls=n_iterations, random_state=42)
    return result.x

# ============================================================================
# MAIN PROCESSING FOR LOL
# ============================================================================

def process_lol_image_improved(low_path, high_path, image_name, show_steps=True):
    low_img = cv2.imread(low_path)
    high_img = cv2.imread(high_path)

    if low_img is None or high_img is None:
        print(f"Error: Could not read {image_name}")
        return None

    print(f"\nProcessing LOL image: {image_name}")
    
    try:
        denoised_low_img = adaptive_denoising(low_img)
        
        # Analyze characteristics
        ifg_preprocessor = AdaptiveIFGForLOL()
        img_characteristics = ifg_preprocessor.analyze_lol_characteristics(denoised_low_img)
        
        # IFG pre-processing
        adapted_generator = ifg_preprocessor.adapt_for_lol(img_characteristics)
        ifg_enhanced = adapted_generator.apply_ifg_preprocessing(denoised_low_img)
        ifg_enhanced = cv2.convertScaleAbs(ifg_enhanced, alpha=1.1, beta=5)

        # Create confidence map from IFG-enhanced image
        saturation_blend_map = compute_saturation_confidence_map(ifg_enhanced)
        print(f"   - Saturation confidence: Min={np.min(saturation_blend_map):.2f}, "
              f"Mean={np.mean(saturation_blend_map):.2f}, Max={np.max(saturation_blend_map):.2f}")

        print(f"   - LOL IFG: Mean={img_characteristics['mean_intensity']:.3f}, "
              f"Dark_ratio={img_characteristics['dark_ratio']:.3f}")
        
        # Optimize parameters
        optimal_params = improved_bayesian_optimization_for_lol(
            ifg_enhanced, high_img, saturation_blend_map, n_iterations=20
        )
        sigma_small_opt, sigma_med_opt, sigma_large_opt, guided_r_opt, guided_eps_opt, contrast_opt, gamma_alpha_opt = optimal_params
        
        hsv_enhanced = cv2.cvtColor(ifg_enhanced, cv2.COLOR_BGR2HSV)
        h_channel = hsv_enhanced[:, :, 0].astype(np.float32)
        s_channel = hsv_enhanced[:, :, 1].astype(np.float32)
        v_channel = hsv_enhanced[:, :, 2].astype(np.float32)

        # Denoise & normalize chromatic components
        h_channel = cv2.bilateralFilter(h_channel, d=5, sigmaColor=25, sigmaSpace=15)
        s_channel = cv2.bilateralFilter(s_channel, d=5, sigmaColor=25, sigmaSpace=15)
        s_channel = normalize_image(s_channel, low_percent=5, high_percent=95)

        # SYNCHRONIZED LOGIC - Apply blend map BEFORE MSR
        s_channel_suppressed = s_channel * saturation_blend_map

        # MSR for V and suppressed S
        reflectance_v = get_msr_reflectance(v_channel, sigmas=[sigma_small_opt, sigma_med_opt, sigma_large_opt])
        reflectance_s = get_msr_reflectance(
            s_channel_suppressed,
            sigmas=[sigma_small_opt/2, sigma_med_opt/2, sigma_large_opt/2]
        )

        # CRITICAL FIX: Scale reflectance to [0, 255] range
        reflectance_s_scaled = np.clip((reflectance_s - 0.5) * 255, 0, 255)

        illumination = get_gaussian_blur(v_channel + 1.0, sigma=sigma_large_opt)
        illumination = adaptive_gamma_correction(illumination, alpha=gamma_alpha_opt)

        reflectance_v = xip.guidedFilter(
            reflectance_v.astype(np.float32),
            reflectance_v.astype(np.float32),
            radius=int(guided_r_opt), eps=guided_eps_opt
        )

        reflectance_v = reflectance_v * contrast_opt
        enhanced_v_channel = reconstruct_image_msr(illumination, reflectance_v)

        # CRITICAL FIX: Use scaled reflectance (NOT raw reflectance)
        enhanced_s_channel = 0.8 * s_channel_suppressed + 0.2 * reflectance_s_scaled

        # Recombine channels
        hsv_enhanced[:, :, 0] = h_channel.astype(np.uint8)
        hsv_enhanced[:, :, 1] = enhanced_s_channel.astype(np.uint8)
        hsv_enhanced[:, :, 2] = enhanced_v_channel.astype(np.uint8)

        final_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        print(f"   - Optimized Parameters: σs={sigma_small_opt:.1f}, σm={sigma_med_opt:.1f}, σl={sigma_large_opt:.1f}, "
              f"r={guided_r_opt:.1f}, ε={guided_eps_opt:.4f}, α={gamma_alpha_opt:.3f}")
        
    except Exception as e:
        print(f"   - Enhancement failed: {e}. Printing traceback...")
        traceback.print_exc()
        final_enhanced = low_img.copy()
        ifg_enhanced = low_img.copy()
        illumination = np.zeros_like(cv2.cvtColor(low_img, cv2.COLOR_BGR2GRAY))
        reflectance_v = np.zeros_like(illumination)
        img_characteristics = {'mean_intensity': 0, 'dark_ratio': 0}

    enhanced_gray = cv2.cvtColor(final_enhanced, cv2.COLOR_BGR2GRAY)
    high_gray = cv2.cvtColor(high_img, cv2.COLOR_BGR2GRAY)
    low_gray = cv2.cvtColor(low_img, cv2.COLOR_BGR2GRAY)
    
    ssim_score = compute_ssim_fixed(high_gray, enhanced_gray)
    psnr_score = psnr(high_img, final_enhanced)
    
    enhanced_grad = np.gradient(enhanced_gray)
    high_grad = np.gradient(high_gray)
    grad_correlation = np.corrcoef(
        np.sqrt(enhanced_grad[0]**2 + enhanced_grad[1]**2).flatten(),
        np.sqrt(high_grad[0]**2 + high_grad[1]**2).flatten()
    )[0,1]
    grad_correlation = np.nan_to_num(grad_correlation, nan=0.0)
    
    improvement_factor = np.mean(enhanced_gray) / (np.mean(low_gray) + 1e-8)
    
    print(f"   - Quality Assessment:")
    print(f"       PSNR: {psnr_score:.2f} dB")
    print(f"       SSIM: {ssim_score:.4f}")
    print(f"       Gradient Correlation: {grad_correlation:.3f}")
    print(f"       Brightness Improvement: {improvement_factor:.2f}x")
    
    if show_steps:
        visualize_lol_results(low_img, ifg_enhanced, high_img, final_enhanced, 
                              illumination, reflectance_v, image_name)
    
    return {
        "Image Name": image_name,
        "PSNR": psnr_score,
        "SSIM": ssim_score,
        "Gradient_Correlation": grad_correlation,
        "Improvement_Factor": improvement_factor,
        "LOL_Mean_Intensity": img_characteristics.get('mean_intensity', 0),
        "LOL_Dark_Ratio": img_characteristics.get('dark_ratio', 0),
        "low_img": low_img,
        "ifg_enhanced": ifg_enhanced,
        "high_img": high_img,
        "final_enhanced": final_enhanced,
        "name": os.path.splitext(image_name)[0]
    }

# ============================================================================
# VISUALIZATION FOR LOL
# ============================================================================

def visualize_lol_results(low_img, ifg_enhanced, high_img, final_enhanced, 
                          illumination, reflectance, image_name, save_dir="lol_results_msr"):
    
    os.makedirs(save_dir, exist_ok=True)
    
    low_rgb = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
    ifg_rgb = cv2.cvtColor(ifg_enhanced, cv2.COLOR_BGR2RGB)
    high_rgb = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final_enhanced, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Improved LOL Enhancement (MSR): {image_name}', fontsize=16, fontweight='bold')
    
    axes[0, 0].imshow(low_rgb)
    axes[0, 0].set_title('Original Low-Light', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ifg_rgb)
    axes[0, 1].set_title('Improved IFG', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(high_rgb)
    axes[0, 2].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(final_rgb)
    axes[0, 3].set_title('Final Enhanced (Confidence-Based)', fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(illumination, cmap='gray')
    axes[1, 0].set_title('Illumination Map', fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(reflectance, cmap='gray')
    axes[1, 1].set_title('Reflectance Map', fontsize=10)
    axes[1, 1].axis('off')
    
    axes[1, 2].hist(cv2.cvtColor(low_img, cv2.COLOR_BGR2GRAY).flatten(), 
                    bins=50, alpha=0.7, label='Original', color='blue')
    axes[1, 2].hist(cv2.cvtColor(final_enhanced, cv2.COLOR_BGR2GRAY).flatten(), 
                    bins=50, alpha=0.7, label='Enhanced', color='red')
    axes[1, 2].set_title('Intensity Histograms', fontsize=10)
    axes[1, 2].legend()
    axes[1, 2].set_xlabel('Intensity')
    axes[1, 2].set_ylabel('Frequency')
    
    enhanced_gray = cv2.cvtColor(final_enhanced, cv2.COLOR_BGR2GRAY)
    high_gray = cv2.cvtColor(high_img, cv2.COLOR_BGR2GRAY)
    
    ssim_score = compute_ssim_fixed(high_gray, enhanced_gray)
    psnr_score = psnr(high_img, final_enhanced)
    
    metrics_text = f"Quality Metrics:\n"
    metrics_text += f"PSNR: {psnr_score:.2f} dB\n"
    metrics_text += f"SSIM: {ssim_score:.4f}\n"
    metrics_text += f"Mean Original: {np.mean(cv2.cvtColor(low_img, cv2.COLOR_BGR2GRAY)):.1f}\n"
    metrics_text += f"Mean Enhanced: {np.mean(enhanced_gray):.1f}\n"
    metrics_text += f"Improvement: {np.mean(enhanced_gray)/np.mean(cv2.cvtColor(low_img, cv2.COLOR_BGR2GRAY)):.2f}x"
    
    axes[1, 3].text(0.1, 0.5, metrics_text, fontsize=10,
                    verticalalignment='center', transform=axes[1, 3].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 3].set_title('Metrics', fontsize=10)
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{os.path.splitext(image_name)[0]}_msr_fixed_final.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Results saved: {save_path}")

# ============================================================================
# MAIN EXECUTION FOR LOL
# ============================================================================

def main_lol_enhancement(high_dir, low_dir, max_images=10, show_individual_steps=True):
    if not os.path.exists(low_dir) or not os.path.exists(high_dir):
        print("Dataset directories not found. Please update the paths.")
        return
    
    results = []
    processed_count = 0
    
    print(f"Processing LOL Dataset with Multi-Scale Retinex (MSR) Enhancement...")
    print("✓ MSR for improved structure preservation")
    print("✓ LOL-specific IFG parameter adaptation")
    print("✓ Spatially-adaptive saturation confidence map")
    print("✓ Fixed scale matching for S-channel MSR blending")
    print("✓ Synchronized optimizer and final-pass logic\n")
    
    for img_name in sorted(os.listdir(low_dir)):
        if processed_count >= max_images:
            break
            
        low_path = os.path.join(low_dir, img_name)
        high_path = os.path.join(high_dir, img_name)
        
        if os.path.exists(low_path) and os.path.exists(high_path):
            result = process_lol_image_improved(
                low_path, high_path, img_name,
                show_steps=show_individual_steps
            )
            if result:
                results.append(result)
                processed_count += 1
        else:
            print(f"Skipping {img_name} - File missing.")
    
    if not results:
        print("No images processed successfully.")
        return
    
    avg_psnr = np.mean([r['PSNR'] for r in results])
    avg_ssim = np.mean([r['SSIM'] for r in results])
    avg_grad_corr = np.mean([r['Gradient_Correlation'] for r in results])
    avg_improvement = np.mean([r['Improvement_Factor'] for r in results])
    
    print("\n" + "="*80)
    print("IMPROVED LOL ENHANCEMENT - FINAL VERSION (ALL FIXES APPLIED)")
    print("="*80)
    print(f"Images processed: {len(results)}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average Gradient Correlation: {avg_grad_corr:.3f}")
    print(f"Average Brightness Improvement: {avg_improvement:.2f}x")
    
    csv_results = []
    for result in results:
        csv_result = {
            "Image Name": result["Image Name"],
            "PSNR": result["PSNR"], 
            "SSIM": result["SSIM"],
            "Gradient_Correlation": result["Gradient_Correlation"],
            "Improvement_Factor": result["Improvement_Factor"],
            "LOL_Mean_Intensity": result["LOL_Mean_Intensity"],
            "LOL_Dark_Ratio": result["LOL_Dark_Ratio"]
        }
        csv_results.append(csv_result)
    
    df = pd.DataFrame(csv_results)
    
    print("\nDetailed Results:")
    print(df.to_string(index=False))
    
    df.to_csv("improved_lol_enhancement_msr_results_FINAL.csv", index=False)
    print(f"\nResults saved to: improved_lol_enhancement_msr_results_FINAL.csv")
    
    return results

if __name__ == "__main__":
    HIGH_DIR = r"/Users/tharunbalasubramaniam/Downloads/Dataset_Neural_Network/Tharun/LOLdataset/our485/high"
    LOW_DIR = r"/Users/tharunbalasubramaniam/Downloads/Dataset_Neural_Network/Tharun/LOLdataset/our485/low"
    
    MAX_IMAGES = 487
    SHOW_INDIVIDUAL_STEPS = True
    
    results = main_lol_enhancement(
        high_dir=HIGH_DIR,
        low_dir=LOW_DIR,
        max_images=MAX_IMAGES,
        show_individual_steps=SHOW_INDIVIDUAL_STEPS
    )