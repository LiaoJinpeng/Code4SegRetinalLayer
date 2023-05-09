# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:12:50 2022

@author: JINPENG LIAO

-	Functions:
1.	Define the loss function for network training (Input: Validation&Output)
2.	Define the Optimizer of network training
3.	Define the Network Training Type: 
    a) Supervised 
    b) Semi-supervised , Currently not support
    c) Unsupervised
4.  Define the Network Architecture (Include Return Networks)

"""
# %% System Setup
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")


# %%  AIO Unsupervised Training Pipeline
class UnsupervisedTrain(tf.keras.Model):
    def __init__(self, discriminator, generator, g_loss_weight=1e-3):
        super(UnsupervisedTrain, self).__init__()
        self.D = discriminator
        self.G = generator
        self.factor = (1 / g_loss_weight)

    def compile(self, d_optimizer, g_optimizer,
                d_loss_function, g_loss_function, s_loss_function,
                metrics_fn=None):
        super(UnsupervisedTrain, self).compile()

        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        self.D_loss = d_loss_function
        self.G_loss = g_loss_function
        self.S_loss = s_loss_function

        self.metrics_fn = metrics_fn

    @tf.function
    def test_step(self, datasets):
        valid_input, valid_true = datasets
        valid_pred = self.G(valid_input, training=False)

        for i in range(len(self.metrics_fn)):
            self.metrics_fn[i].update_state(valid_true, valid_pred)

        return {m.name: m.result() for m in self.metrics_fn}

    @tf.function
    def train_step(self, datasets):
        train_input, train_true = datasets

        with tf.GradientTape(persistent=True) as tape:
            train_pred = self.G(train_input, training=True)

            real_logit = self.D(train_true, training=True)
            fake_logit = self.D(train_pred, training=True)

            d_loss = self.D_loss(real_out=real_logit, fake_out=fake_logit)
            g_loss = self.G_loss(real_out=real_logit, fake_out=fake_logit)

            s_loss = self.S_loss(train_pred, train_true)
            t_loss = s_loss + g_loss

        d_variables = self.D.trainable_variables
        g_variables = self.G.trainable_variables

        d_grads = tape.gradient(d_loss, d_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, d_variables))

        g_grads = tape.gradient(t_loss, g_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, g_variables))

        return {
            "DLoss": d_loss,
            "GLoss": g_loss * self.factor,
            "SLoss": s_loss,
        }


# %% AIO Supervised Training Pipeline
class SupervisedTrain(tf.keras.Model):
    def __init__(self, model=None):
        super(SupervisedTrain, self).__init__()
        self.model = model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def compile(self, optimizers, loss_function, metrics_fn=None):
        super(SupervisedTrain, self).compile()
        self.optimizer = optimizers
        self.loss_fn = loss_function
        self.metrics_fn = metrics_fn

    @tf.function
    def test_step(self, datasets):
        for i in range(len(self.metrics_fn)):
            self.metrics_fn[i].reset_state()

        valid_input, valid_true = datasets
        valid_pred = self.model(valid_input, training=False)

        for i in range(len(self.metrics_fn)):
            self.metrics_fn[i].update_state(valid_true, valid_pred)

        return {m.name: m.result() for m in self.metrics_fn}

    @tf.function
    def train_step(self, datasets):
        train_input, train_label = datasets

        with tf.GradientTape() as tape:
            train_pred = self.model(train_input, training=True)
            loss = self.loss_fn(train_pred, train_label)

        variables = self.model.trainable_variables
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        return {"Loss:": loss}


# %% AIO Training Pipeline for Diffusion Type Model
class TrainDDIM(tf.keras.Model):

    def __init__(self, network=None, width=128, batch_size=64,
                 max_signal_rate=0.95, min_signal_rate=0.02):
        super().__init__()

        self.normalizer = tf.keras.layers.Normalization()
        self.network = network
        self.ema_network = tf.keras.models.clone_model(network)

        self.max_signal_rate = max_signal_rate
        self.min_signal_rate = min_signal_rate

        self.width = width  # width = height of the input data shape
        self.batch_size = batch_size

    def compile(self, KID, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = tf.keras.metrics.Mean(name='noise_loss')
        self.image_loss_tracker = tf.keras.metrics.mean(name='image_loss')
        self.kid = KID

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range.
        images = self.normalizer.mean + images * self.normalizer.variance ** 0.5
        return tf.clip_by_value(images, 0., 1.)

    def diffusion_schedule(self, diffusion_times):
        """ We need to have a function that tells us at each point in the
        diffusion process the noise levels and signal levels of the noisy image
        corresponding to the actual diffusion time.

        We generate the noisy image by weighting the random noise and the
        training image by their corresponding rates and adding them together.

        Squares of their rates can be interpreted as their variance.

        The rates will always be set so that their squared sum is 1, meaning
        that the noisy images will always have unit variance, just like its
        unscaled components.

        """
        # diffusion times -> angles
        start_angle = tf.acos(self.max_signal_rate)  # 0.95
        end_angle = tf.acos(self.min_signal_rate)  # 0.02

        diffusion_angles = start_angle + diffusion_times * (
                end_angle - start_angle)

        # angles -> signal and noise rates (squares of them is 1)
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average (ema) weights are used at test-step
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it.
        pred_noises = network([noisy_images, noise_rates ** 2],
                              training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        """ 1. We take the previous estimate of the noisy image and separate it
        into image and noise using our network. 2. Then, we recombine these
        components using the signal and noise rate of the following step.

        This example only implements the deterministic sampling procedure from
        DDIM, which corresponds to eat = 0 in the paper. Stochastic sampling can
        be used without retraining the network (since both models are trained
        the same way), and it can improve sample quality, while on the other
        hand requiring more sampling steps usually.

        """
        # reverse diffusion -- sampling the image from the noise
        num_images = initial_noise.shape[0]
        step_size = 1. / diffusion_steps

        # at the first sampling step, the 'noisy image' is pure noise, but its
        # signal rate is assumed to be non-zero (i.e., min signal rate).
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components (i.e., images
            # and noise for neural network input). Diffusion time is a numerical
            # parameter between 0 and 1 to control the rates of signal and noise
            # , where signal rate + noise rate = 1.
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size

            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

            pred_noises, pred_images = self.denoise(  # use for evaluation step
                noisy_images, noise_rates, signal_rates, training=False
            )

            # remix the predicted component using the next signal and noise rate
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                    next_signal_rates * pred_images +
                    next_noise_rates * pred_noises
            )  # this new noisy image will be used in the next step.

        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(
            shape=(num_images, self.width, self.width, 3))
        generated_images = self.reverse_diffusion(
            initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, datasets):
        # TODO: Unfinished code for DDIM. Reason: cannot use the tar_label for
        #  neural network training
        input_image, tar_label = datasets
        ema = 0.999

        # normalize image to have standard deviation of 1, like noises
        input_image = self.normalizer(input_image, training=True)
        noises = tf.random.normal(
            shape=(self.batch_size, self.width, self.width, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(shape=(self.batch_size, 1, 1, 1),
                                            minval=0., maxval=1.)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        # mix the images with noises accordingly
        noisy_images = signal_rates * input_image + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)
            image_loss = self.loss(input_image, pred_images)

        gradients = tape.gradient(
            noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(
            gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights,
                                      self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase
        return {m.name: m.result() for m in self.metrics[:-1]}


# %% DDPM Relatively Function/Class
class GaussianDiffusion:
    """ Linear Gaussian Diffusion.
    Define the forward process and the reverse process as a separate utility.
    Most of the code in this utility has been borrowed from the original
    implementation with some slight modifications.

    (0) Description of the variables or terms

    - example of cumprod: if a = [1, 2, 4], np.cumprod(a) = [1, 1*2, 1*2*4]

    - beta: or called beta_t in diffusion model, it is a hyperparameter changed
    with the current timestep (t) in the forward/reverse process.

    -

    (1) Description of the forward process

    - Calculate the t-step noisy image x_t based on the x0 and 1 - alpha_cumprod
        x_t = √(alpha_cumprod_t) * x_0 + √(1 - alpha_cumprod_t) * ε_cumprod_t
        where alpha_cumprod_t = [1, a1, a1*a2, a1*a2*a3, ..., a1*...*a_{t-1}],
        ε_cumprod_t ~ N(0, I), and alpha_cumprod_t = 1 - beta_cumprod_t

    - Calculate the t-step noisy image x_t based on the previous x_{t-1} image.
        x_t = √(alpha_t) * x_{t-1} + √(1 - alpha_t) * ε_t
        where alpha_t = 1 - beta_t, and ε_t ~ N(0, I)


    """

    def __init__(self, timestep=1000, clip_min=0., clip_max=1.,
                 beta_start=1e-4, beta_end=0.02):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timestep = timestep
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Define the linear variance schedule.
        self.betas = betas = np.linspace(
            beta_start, beta_end, timestep, dtype=np.float64)
        self.timestep = int(timestep)

        alphas = 1. - betas
        # alphas_cumprod = [a1, a1*a2, a1*a2*a3, ..., a1*...*at]
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # alphas_cumprod_prev = [1, a1, a1*a2, a1*a2*a3, ..., a1*...*a_{t-1}]
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        # convert to tf variables.
        typed = tf.float32  # type of data
        self.betas = tf.constant(betas, dtype=typed)
        self.alpha_cumprod = tf.constant(alphas_cumprod, dtype=typed)
        self.alpha_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=typed)

        # calculate for the diffusion (forward process) steps mentioned in (1)
        self.sqrt_alphas_cumprod = tf.constant(
            np.sqrt(alphas_cumprod), dtype=typed)  # √(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.constant(
            np.sqrt(1. - alphas_cumprod), dtype=typed)  # √(1 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = tf.constant(
            np.log(1. - alphas_cumprod), dtype=typed)  # log(1 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = tf.constant(
            np.sqrt(1. / alphas_cumprod), dtype=typed)  # √(1 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = tf.constant(
            np.sqrt(1. / alphas_cumprod - 1), dtype=typed
        )  # √((1 / alphas_cumprod) - 1)

        # calculate for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
                betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_variance = tf.constant(posterior_variance, dtype=typed)

        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = tf.constant(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype=typed)
        self.posterior_mean_coef1 = tf.constant(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod),
            dtype=typed)
        self.posterior_mean_coef2 = tf.constant(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)
            , dtype=typed)

    def _extract(self, a, t, x_shape):
        """ Extract the coefficients at the specific timestep, then reshape to
        [batch_size, 1, 1, 1] for broadcasting purpose.
        """
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1, 1])

    def q_mean_variance(self, x_start, t):
        """ Extract the mean, and the variance at current timestep.

        Mathmatics Equation:
            xt~q(xt|x0) = N(xt; √(alphas_cumprod)x0, (1 - alphas_cumprod)I)

        where √(alphas_cumprod)x0 is the mean of the gaussian noise,
        (1-alphas_cumprod) is the variance of the gaussian noise.
        both of them are coefficients for q_sample calculation.
        """
        x_start_shape = tf.shape(x_start)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape
                             ) * x_start  # x_start is the x0.
        variance = self._extract(1. - self.alpha_cumprod, t, x_start_shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start_shape)

        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise):
        """ Diffusion step for image, also called forward process.
        Mathmatics Equation:
            xt = √(alphas_cumprod) * x0 + √(1 - alphas_cumprod) * ε, ε ~ N(0, I)

        :param x_start: initial sample (before the first diffusion step)
        :param t: current timestep
        :param noise: gaussian noise to be added at the current timestep

        :return: diffused samples at timestep (t).

        """
        xshape = tf.shape(x_start)
        return (
                self._extract(self.sqrt_alphas_cumprod, t, xshape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod, t,
                              xshape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """ Compute the mean and variance of the diffusion
        posterior q(x_{t-1} | x_t, x_0)

        Mathmatics Equation:
            q(x_{t-1} | x_t, x_0) = mean_coef1 * x0 + mean_coef2 * xt + variance
        where:
            mean_coef1 = (√(alpha_cumprod[t-1]) * β[t]) / (1 - alpha_cumprod[t])
            mean_coef2 = (√(alpha[t]) * (1 - alpha_cumprod[t-1])) / (
                1 - alpha_cumprod[t])
            variance = β[t] * ((1 - alpha_cumprod[t-1]) / (1 - alpha_cumprod))
            β[t] = 1 - alpha[t], a coefficient decide by time step (t)

        Args:
            x_start: the starting point (x0) for the posterior computation, in
            this function, the x_start is calculated based on the pred_noise
            (from neural network) by func: 'predict_start_from_noise'.
            x_t: the sample of image at timestep t.
            t: timestep t

        Returns: posterior mean and variance at current timestep

        """
        xshape = tf.shape(x_t)
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, xshape) * x_start +
                self._extract(self.posterior_mean_coef2, t, xshape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, xshape)

        posterior_log_variance_clip = self._extract(
            self.posterior_log_variance_clipped, t, xshape)

        return posterior_mean, posterior_variance, posterior_log_variance_clip

    def predict_start_from_noise(self, x_t, t, noise):
        """ Use x_t and noise to represent the x_start (x0).
        Link: https://www.cvmart.net/community/detail/6936

        Details Description: based on the equation 'q_sample', we can convert to
        another format to find the x_start (x0) based on the x_t and timestep t.
        And that mathmatics equation: is:
            (x_t - √(1 - alpha_cumprod[t]) * noise) / √(alpha_cumprod[t]).
        We do some change on the above equation, and it will be:
            x_start = (1 / √(alpha_cumprod[t])) * x_t - (
                √(1 - alpha_cumprod[t]) / √(alpha_cumprod[t])) * noise

        Args:
            x_t: the x_t at the timestep (t)
            t: the timestep (t)
            noise: the predicted noise from the trained neural network as output

        Returns: the x_0 (x_start) image sampled from the noise

        """
        xshape = tf.shape(x_t)
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, xshape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t,
                              xshape) * noise
        )

    def p_mean_variance(self, pred_noise, x, t, use_clip=True):
        # the x_t is begun from random gaussian noise here, t is the time step,
        # pred_noise is calculate from the neural network.
        x_recon = self.predict_start_from_noise(x_t=x, t=t, noise=pred_noise)
        if use_clip:
            x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)

        # posterior: q(x_{t-1} | x_t, x_0), use x_t and x_0 to calculate x_{t-1}
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_recon, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, use_clip=True):
        # based on the next mean and variance predict by func 'q_posterior',
        # re-sample the image from the noise.
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x, t, use_clip)

        model_variance = tf.exp(0.5 * model_log_variance)

        noise = tf.random.normal(shape=x.shape, dtype=x.dtype)
        nonzero_mask = tf.reshape(
            1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1, 1])

        return model_mean + model_variance * noise * nonzero_mask


class TrainDDPM(tf.keras.Model):
    """ Diffusion Model Training

    We follow the same setup for training the diffusion model as describe in the
    paper. We use Adam optimizer with a learning rate of 2e-4. We use EMA on
    model parameters with a decay factor of 0.999.

    We treat our model as noise prediction network. i.e., at every training step
    we input a batch of images and corresponding time steps to our network, and
    the network outputs the noise as prediction.

    Note: We are using mean squared error as the loss function which is aligned
    with the paper, and theoretically makes sense. In practice, though, it is
    also common to use mean absolute error or Huber loss as the loss function.

    """

    def __init__(self, network, ema_network, imshape, timestep=1000, ema=0.999):
        super().__init__()

        self.network = network
        self.ema_network = ema_network

        self.timestep = timestep
        self.ema = ema

        self.gdf_util = GaussianDiffusion(
            timestep=timestep, clip_min=0., clip_max=1.,
            beta_start=1e-4, beta_end=0.02)

        self.img_size = imshape[1]
        self.img_channel = imshape[-1]

    def train_step(self, datasets):

        train_input, train_valid = datasets

        # 1. Get the batch size
        batch_size = tf.shape(train_input)[0]

        # 2. Sample timestep uniformly
        t = tf.random.uniform(
            minval=0, maxval=self.timestep, shape=(batch_size,), dtype=tf.int64
        )

        with tf.GradientTape() as tape:
            # 3. sample random noise to be added to the image in the batch
            noise = tf.random.normal(shape=tf.shape(train_input),
                                     dtype=train_input.dtype)

            # 4. diffuse the images with noise, and obtain x_t from t and x_0
            x_t = self.gdf_util.q_sample(train_input, t, noise)

            # 5. pass the diffused images and time steps to the network
            pred_noise = self.network([x_t, t], training=True)

            # 6. calculate the noise between pred_noise and noise
            loss = self.loss(noise, pred_noise)

        # 7. update the weights for the neural network
        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights))

        # 8. update the weights for ema_network. EMA: exponential moving average
        # to prevent over fitting and better validation result.
        for weight, ema_weight in zip(
                self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return {'n_loss': loss}

    def generate_images(self, num_images=16):
        # 1. randomly sample noise (starting point of the reverse process)
        samples = tf.random.normal(
            shape=(num_images, self.img_size, self.img_size, self.img_channel),
            dtype=tf.float32
        )

        # 2. sample from the model iteratively
        for t in reversed(range(0, self.timestep)):
            # 2.1 obtain the current t.
            t_now = tf.cast(tf.fill(num_images, t), dtype=tf.int64)

            # 2.2 predict the noise from neural network.
            pred_noise = self.ema_network.predict(
                [samples, t_now], verbose=0, batch_size=num_images)

            # 2.3 use the pred_noise, x_t, and t_now to generate the images.
            samples = self.gdf_util.p_sample(pred_noise, samples, t_now)

        return samples

    def plot_images(self, epoch=None, logs=None, num_rows=2, num_cols=8):
        generated_samples = self.generate_images(num_images=num_rows * num_cols)

        generated_samples = tf.clip_by_value(generated_samples * 255., 0., 255.)
        generated_samples = generated_samples.numpy().astype(np.uint8)

        _, ax = plt.subplots(num_rows, num_cols, figsize=(12, 5))
        for i, image in enumerate(generated_samples):
            if num_rows == 1:
                ax[i].imshow(image)
                ax[i].axis("off")
            else:
                ax[i // num_cols, i % num_cols].imshow(image)
                ax[i // num_cols, i % num_cols].axis("off")

        plt.tight_layout()
        plt.show()
