import numpy as np
import theano
import theano.tensor as T

from ..utils import actfuncs, costfuncs
from ..utils.math import nprng, tnrng


class RBM(object):

    def __init__(self, input, vshape, hshape,
                 W=None, vbias=None, hbias=None):
        super(RBM, self).__init__()

        self._input = input

        if isinstance(vshape, int):
            self._vshape = vshape
        elif isinstance(vshape, tuple) or isinstance(vshape, list):
            self._vshape = np.prod(vshape)
        else:
            raise ValueError("vshape type error")

        if isinstance(hshape, int):
            self._hshape = hshape
        elif isinstance(hshape, tuple) or isinstance(hshape, list):
            self._hshape = np.prod(hshape)
        else:
            raise ValueError("hshape type error")

        # Initialize parameters
        if W is None:
            W_bound = np.sqrt(6.0 / (self._vshape + self._hshape))
            W_bound *= 4

            init_W = np.asarray(
                nprng.uniform(low=-W_bound, high=W_bound,
                              size=(self._vshape, self._hshape)),
                dtype=theano.config.floatX)

            self._W = theano.shared(value=init_W, borrow=True)
        else:
            self._W = W

        if vbias is None:
            init_vbias = np.zeros(
                self._vshape, dtype=theano.config.floatX)

            self._vbias = theano.shared(value=init_vbias, borrow=True)

        if hbias is None:
            init_hbias = np.zeros(
                self._hshape, dtype=theano.config.floatX)

            self._hbias = theano.shared(value=init_hbias, borrow=True)

        self._params = [self._W, self._vbias, self._hbias]

    def free_energy(self, v_sample):
        z = T.dot(v_sample, self._W) + self._hbias
        vbias_term = T.dot(v_sample, self._vbias)
        hidden_term = T.log(1 + T.exp(z)).sum(axis=1)
        return -hidden_term - vbias_term

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1 = T.dot(v0_sample, self._W) + self._hbias
        h1_mean = actfuncs.sigmoid(pre_sigmoid_h1)
        h1_sample = tnrng.binomial(size=h1_mean.shape, n=1, p=h1_mean,
                                   dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1 = T.dot(h0_sample, self._W.T) + self._vbias
        v1_mean = actfuncs.sigmoid(pre_sigmoid_v1)
        v1_sample = tnrng.binomial(size=v1_mean.shape, n=1, p=v1_mean,
                                   dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def get_cost_updates(self, lr=1e-3, persistent=None, k=1):
        if persistent is None:
            __, __, chain_start = self.sample_h_given_v(self._input)
        else:
            chain_start = persistent

        init_outputs = [None] * 5 + [chain_start]

        [pre_sigmoid_nvs, nv_means, nv_samples,
         pre_sigmoid_nhs, nh_means, nh_samples], updates = \
            theano.scan(self.gibbs_hvh, outputs_info=init_outputs, n_steps=k)

        chain_end = nv_samples[-1]

        cost = self.free_energy(self._input).mean() - \
            self.free_energy(chain_end).mean()

        grads = T.grad(cost, self._params, consider_constant=[chain_end])
        for p, g in zip(self._params, grads):
            updates[p] = p - g * T.cast(lr, dtype=theano.config.floatX)

        if persistent:
            updates[persistent] = nh_samples[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        bit_i_idx = theano.shared(value=0)

        xi = T.round(self._input)
        fe_xi = self.free_energy(xi)

        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        fe_xi_flip = self.free_energy(xi_flip)

        cost = T.mean(self._vshape * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

        updates[bit_i_idx] = (bit_i_idx + 1) % self._vshape

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        y = actfuncs.sigmoid(pre_sigmoid_nv)
        return costfuncs.binxent(y, self._input)
