from .spectral_rnn.spectral_rnn import SpectralRNN, SpectralRNNConfig
from .spectral_rnn.manifold_optimization import ManifoldOptimizer
from .spectral_rnn.cgRNN import clip_grad_value_complex_
from .transformer import TransformerPredictor, TransformerConfig
from .maf import MAFEstimator
from .cwspn import CWSPN, CWSPNConfig
from .model import Model

import numpy as np
import torch
import torch.nn as nn
from darts.darts_cnn.model_search import Network as CWSPNModelSearch
from darts.darts_rnn.model_search import RNNModelSearch
from rtpt import RTPT
# Use GPU if avaiable
device = 'cuda:5' if torch.cuda.is_available() else 'cpu'

def load_arch_params(srnn=True):

    if srnn:
        arch_params_raw = np.array([[0.05608100816607475, 0.06870987266302109, 0.2165592759847641, 0.08869156241416931, 0.5699583292007446], [0.06695377081632614, 0.1252291202545166, 0.13493241369724274, 0.24823960661888123, 0.4246450960636139], [0.05238443613052368, 0.12590470910072327, 0.23995350301265717, 0.08084835857152939, 0.5009090304374695], [0.1089782863855362, 0.11746864020824432, 0.1312309354543686, 0.4640672206878662, 0.1782548874616623], [0.0736430436372757, 0.18587562441825867, 0.16631470620632172, 0.1537150740623474, 0.4204515814781189], [0.05640175938606262, 0.10854099690914154, 0.28099095821380615, 0.07116516679525375, 0.4829011857509613], [0.13197152316570282, 0.14132705330848694, 0.15703606605529785, 0.4055844247341156, 0.16408094763755798], [0.08837749809026718, 0.20374871790409088, 0.17323771119117737, 0.15755560994148254, 0.3770804703235626], [0.0661877989768982, 0.13302665948867798, 0.2434728443622589, 0.08838000893592834, 0.46893271803855896], [0.058981917798519135, 0.07889309525489807, 0.3483126759529114, 0.06881941109895706, 0.44499292969703674], [0.17021507024765015, 0.17539912462234497, 0.1864837408065796, 0.285239040851593, 0.1826629936695099], [0.11772088706493378, 0.22900131344795227, 0.20035812258720398, 0.16958673298358917, 0.2833329439163208], [0.08337496221065521, 0.1708083301782608, 0.223876953125, 0.10805583745241165, 0.41388386487960815], [0.06985853612422943, 0.10462936758995056, 0.315256267786026, 0.08280841261148453, 0.42744743824005127], [0.057064253836870193, 0.0628884881734848, 0.37389230728149414, 0.06065619736909866, 0.4454987049102783], [0.18945087492465973, 0.19281838834285736, 0.196219801902771, 0.2273416519165039, 0.1941693127155304], [0.14507150650024414, 0.22756876051425934, 0.20698902010917664, 0.17830030620098114, 0.24207043647766113], [0.1120440736413002, 0.20287199318408966, 0.22369177639484406, 0.13674162328243256, 0.3246505558490753], [0.09417062252759933, 0.14116065204143524, 0.2745150625705719, 0.11132145673036575, 0.3788321912288666], [0.07204936444759369, 0.08715005218982697, 0.3381374180316925, 0.07863444089889526, 0.4240286648273468], [0.05729006603360176, 0.06034564599394798, 0.3878501355648041, 0.05868527293205261, 0.4358288049697876], [0.188343808054924, 0.20513246953487396, 0.1976127177476883, 0.20339077711105347, 0.20552024245262146], [0.16018912196159363, 0.2242700606584549, 0.2047976851463318, 0.17926950752735138, 0.23147368431091309], [0.13333413004875183, 0.21525612473487854, 0.2200762778520584, 0.15243662893772125, 0.2788967788219452], [0.1152176782488823, 0.16711878776550293, 0.24617604911327362, 0.13118776679039001, 0.34029972553253174], [0.08805697411298752, 0.11109905689954758, 0.29236093163490295, 0.0966021716594696, 0.41188082098960876], [0.06798391044139862, 0.07587527483701706, 0.35315772891044617, 0.07069459557533264, 0.4322885572910309], [0.0568794347345829, 0.05870075523853302, 0.39412784576416016, 0.05741627886891365, 0.43287569284439087], [0.1947139948606491, 0.2032701075077057, 0.2002510279417038, 0.1984744668006897, 0.20329038798809052], [0.17659235000610352, 0.21458548307418823, 0.2054865062236786, 0.18568778038024902, 0.21764793992042542], [0.1577150523662567, 0.21488045156002045, 0.216208815574646, 0.1703067272901535, 0.24088892340660095], [0.14450430870056152, 0.1902054101228714, 0.23421362042427063, 0.15726391971111298, 0.27381274104118347], [0.11698297411203384, 0.14429841935634613, 0.2605959177017212, 0.12642213702201843, 0.3517006039619446], [0.08916231989860535, 0.10330653935670853, 0.31084781885147095, 0.09377823770046234, 0.40290507674217224], [0.07104254513978958, 0.0761086568236351, 0.36366507411003113, 0.07257194817066193, 0.41661176085472107], [0.05563561990857124, 0.0564069002866745, 0.3970334529876709, 0.05589016154408455, 0.43503379821777344]])
        arch_params = (arch_params_raw == arch_params_raw.max(axis=1, keepdims=1)).astype(float)
        arch_params_torch = torch.tensor(arch_params).to(device=device)

    else:
        arch_params_raw = np.array([[0.06384659558534622, 0.5505385994911194, 0.07797644287347794, 0.08145342767238617, 0.12179607897996902, 0.10438880324363708], [0.08440500497817993, 0.5861014723777771, 0.07953676581382751, 0.08157417178153992, 0.08314943313598633, 0.08523315191268921], [0.09188734740018845, 0.545225977897644, 0.08822599053382874, 0.09017692506313324, 0.09174462407827377, 0.09273910522460938], [0.09902982413768768, 0.4721280038356781, 0.09985450655221939, 0.11034899204969406, 0.11892024427652359, 0.09971845895051956], [0.11186032742261887, 0.40993887186050415, 0.11336946487426758, 0.1225142776966095, 0.12998069822788239, 0.11233635991811752], [0.09896858781576157, 0.47487974166870117, 0.09904322028160095, 0.11002866178750992, 0.11776135861873627, 0.09931853413581848], [0.11729684472084045, 0.38718652725219727, 0.11471730470657349, 0.13777273893356323, 0.10945937037467957, 0.13356724381446838], [0.13299253582954407, 0.3077976107597351, 0.13066309690475464, 0.15415938198566437, 0.12554007768630981, 0.14884722232818604], [0.11983159184455872, 0.37255942821502686, 0.11799908429384232, 0.14079411327838898, 0.11198016256093979, 0.13683554530143738], [0.09894577413797379, 0.4744432270526886, 0.09651047736406326, 0.12193871289491653, 0.0907444879412651, 0.11741729080677032], [0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204], [0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204], [0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204], [0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204]])
        arch_params = (arch_params_raw == arch_params_raw.max(axis=1, keepdims=1)).astype(float)
        arch_params_torch = torch.tensor(arch_params).to(device=device)

    return arch_params_torch

class PWN(Model):

    def __init__(self, s_config: SpectralRNNConfig, c_config: CWSPNConfig, train_spn_on_gt=True,
                 train_spn_on_prediction=False, train_rnn_w_ll=False, weight_mse_by_ll=None, always_detach=False,
                 westimator_early_stopping=5, step_increase=False, westimator_stop_threshold=.5,
                 westimator_final_learn=2, ll_weight=0.5, ll_weight_inc_dur=20, use_transformer=False, use_maf=False,
                 smape_target=False):

        assert train_spn_on_gt or train_spn_on_prediction
        assert not train_rnn_w_ll or train_spn_on_gt

        self.srnn = SpectralRNN(s_config) if not use_transformer else TransformerPredictor(
            TransformerConfig(normalize_fft=True, window_size=s_config.window_size,
                              fft_compression=s_config.fft_compression))
        self.westimator = CWSPN(c_config) if not use_maf else MAFEstimator()

        self.train_spn_on_gt = train_spn_on_gt
        self.train_spn_on_prediction = train_spn_on_prediction
        self.train_rnn_w_ll = train_rnn_w_ll
        self.weight_mse_by_ll = weight_mse_by_ll
        self.always_detach = always_detach

        self.westimator_early_stopping = westimator_early_stopping
        self.westimator_stop_threshold = westimator_stop_threshold
        self.westimator_final_learn = westimator_final_learn
        self.ll_weight = ll_weight
        self.ll_weight_inc_dur = ll_weight_inc_dur
        self.step_increase = step_increase
        self.use_transformer = use_transformer
        self.use_maf = use_maf
        self.smape_target = smape_target

    def train(self, x_in, y_in, val_x, val_y, embedding_sizes, batch_size=256, epochs=70, lr=0.004, lr_decay=0.97):

        
        if type(self.srnn) == TransformerPredictor:
            lr /= 10
        elif self.srnn.config.rnn_layer_config.use_cg_cell:
            lr /= 4

        # Model is trained jointly on all groups
        x_ = np.concatenate(list(x_in.values()), axis=0)
        y_ = np.concatenate(list(y_in.values()), axis=0)[:, :, -1]

        if self.srnn.final_amt_pred_samples is None:
            self.srnn.final_amt_pred_samples = next(iter(y_in.values())).shape[1]

        # Init SRNN
        x = torch.from_numpy(x_).float().to(device)
        y = torch.from_numpy(y_).float().to(device)
        self.srnn.config.embedding_sizes = embedding_sizes
        self.srnn.build_net()

        self.westimator.stft_module = self.srnn.net.stft

        if self.srnn.config.use_searched_srnn:
            search_stft = self.srnn.net.stft
            emsize = 300
            nhid = 300
            nhidlast = 300
            ntokens = 10000
            dropout = 0
            dropouth = 0
            dropouti = 0
            dropoute = 0
            dropoutx = 0
            config_layer = self.srnn.config
            srnn_search = RNNModelSearch(search_stft, config_layer, ntokens, emsize, nhid, nhidlast,
                                     dropout, dropouth, dropoutx, dropouti, dropoute).to(device=device)
            # set arch weights according to searched cell
            arch_params = load_arch_params(True)
            srnn_search.fix_arch_params(arch_params)
            # set searched srnn arch as srnn net
            self.srnn.net = srnn_search
        # Init westimator
        westimator_x_prototype, westimator_y_prototype = self.westimator.prepare_input(x[:1, :, -1], y[:1])
        self.westimator.input_sizes = westimator_x_prototype.shape[1], westimator_y_prototype.shape[1]
        self.westimator.create_net()

        if self.westimator.config.use_searched_cwspn:
            in_seq_length = self.westimator.input_sizes[0] * (
                2 if self.westimator.use_stft else 1)  # input sequence length into the WeightNN
            output_length = self.westimator.num_sum_params + self.westimator.num_leaf_params  # combined length of sum and leaf params
            sum_params = self.westimator.num_sum_params
            cwspn_weight_nn_search = CWSPNModelSearch(in_seq_length, output_length, sum_params, layers=1, steps=4).to(device=device)
            self.westimator.weight_nn = cwspn_weight_nn_search


        prediction_loss = lambda error: error.mean()
        ll_loss = lambda out: -1 * torch.logsumexp(out, dim=1).mean()
        if self.smape_target:
            smape_adjust = 2  # Move all values into the positive space
            p_base_loss = lambda out, label: 2 * (torch.abs(out - label) /
                                                  (torch.abs(out + smape_adjust) +
                                                   torch.abs(label + smape_adjust))).mean(axis=1)
        # MSE target
        else:
            p_base_loss = lambda out, label: nn.MSELoss(reduction='none')(out, label).mean(axis=1)

        srnn_parameters = list(self.srnn.net.parameters())
        westimator_parameters = self.westimator.parameters()

        amt_param = sum([p.numel() for p in self.srnn.net.parameters()])
        amt_param_w = sum([p.numel() for p in self.westimator.parameters()])

        srnn_optimizer = ManifoldOptimizer(srnn_parameters, lr, torch.optim.RMSprop, alpha=0.9) \
            if type(self.srnn.config) == SpectralRNNConfig and self.srnn.config.rnn_layer_config.use_cg_cell \
            else torch.optim.RMSprop(srnn_parameters, lr=lr, alpha=0.9)
        westimator_optimizer = torch.optim.Adam(westimator_parameters, lr=1e-4)

        if self.train_rnn_w_ll:
            current_ll_weight = 0
            ll_weight_history = []
            ll_weight_increase = self.ll_weight / self.ll_weight_inc_dur
        elif self.train_spn_on_prediction:
            def ll_loss_pred(out, error):
                return (-1 * torch.logsumexp(out, dim=1) * (error ** -2)).mean() * 1e-4

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=srnn_optimizer, gamma=lr_decay)

        westimator_losses = []
        srnn_losses = []
        srnn_losses_p = []
        srnn_losses_ll = []

        stop_cspn_training = False
        westimator_patience_counter = 0
        westimator_losses_epoch = []

        self.srnn.net.train()

        if hasattr(self.westimator, 'spn'):
            self.westimator.spn.train()
            self.westimator.weight_nn.train()
        else:
            self.westimator.model.train()

        if self.srnn.config.use_cached_predictions:
            import pickle
            with open('srnn_train.pkl', 'rb') as f:
                all_predictions, all_f_cs = pickle.load(f)

                all_predictions = torch.from_numpy(all_predictions).to(device)
                all_f_cs = torch.from_numpy(all_f_cs).to(device)

        val_errors = []
        rt = RTPT('JS', 'Bi-Level Eval', epochs)
        rt.start()
        print(f'Starting Training of {self.identifier} model')
        for epoch in range(epochs):
            idx_batches = torch.randperm(x.shape[0], device=device).split(batch_size)

            #if epoch % 2:
            #    model_base_path = 'res/models/'
            #    model_name = f'{self.identifier}-{str(epoch)}'
            #    model_filepath = f'{model_base_path}{model_name}'
            #    self.save(model_filepath)

            if self.train_rnn_w_ll:
                ll_weight_history.append(current_ll_weight)

            srnn_loss_p_e = 0
            srnn_loss_ll_e = 0

            pi = torch.tensor(np.pi)
            srnn_loss_e = 0
            westimator_loss_e = 0
            for i, idx in enumerate(idx_batches):
                batch_x, batch_y = x[idx].detach().clone(), y[idx, :].detach().clone()
                batch_westimator_x, batch_westimator_y = self.westimator.prepare_input(batch_x[:, :, -1], batch_y)

                if self.srnn.config.use_cached_predictions:
                    batch_p = all_predictions.detach().clone()[idx]
                    batch_fc = all_f_cs.detach().clone()[idx]

                if self.train_spn_on_gt:
                    westimator_optimizer.zero_grad()
                    if not stop_cspn_training or epoch >= epochs - self.westimator_final_learn:
                        out_w, _ = self.call_westimator(batch_westimator_x, batch_westimator_y)

                        if hasattr(self.westimator, 'spn'):
                            gt_ll = out_w
                            westimator_loss = ll_loss(gt_ll)
                            westimator_loss.backward()
                            westimator_optimizer.step()
                        else:
                            if self.westimator.use_made:
                                raise NotImplementedError  # MADE not implemented here

                            u, log_det = out_w

                            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                            negloglik_loss += 0.5 * self.westimator.final_input_sizes * np.log(2 * pi)
                            negloglik_loss -= log_det
                            negloglik_loss = torch.mean(negloglik_loss)

                            negloglik_loss.backward()
                            westimator_loss = negloglik_loss.item()
                            westimator_optimizer.step()
                            westimator_optimizer.zero_grad()

                    else:
                        westimator_loss = westimator_losses_epoch[-1]

                    westimator_loss_e += westimator_loss.detach()

                # Also zero grads for westimator, s.t. old grads dont influence the optimization
                srnn_optimizer.zero_grad()
                westimator_optimizer.zero_grad()

                if not self.srnn.config.use_cached_predictions:
                    if self.srnn.config.use_searched_srnn:
                        prediction, f_c = self.srnn.net(batch_x, batch_y, self.srnn.net.weights,
                                                        return_coefficients=True)
                    else:
                        prediction, f_c = self.srnn.net(batch_x, batch_y, return_coefficients=True)
                else:
                    prediction, f_c = batch_p, batch_fc

                if self.westimator.use_stft:
                    prediction_ll, w_in = self.call_westimator(batch_westimator_x, f_c.reshape((f_c.shape[0], -1))
                                        if self.train_rnn_w_ll and not self.always_detach else f_c.reshape((f_c.shape[0], -1)).detach())
                else:
                    prediction_ll, w_in = self.call_westimator(batch_westimator_x, prediction
                                        if self.train_rnn_w_ll and not self.always_detach else prediction.detach())

                #if type(self.srnn) == TransformerPredictor:
                #    num_to_cut = int(prediction_raw.shape[1]/3)
                #    prediction = prediction_raw[:,num_to_cut:]
                #else:
                #    prediction = prediction_raw

                error = p_base_loss(prediction, batch_y)
                p_loss = prediction_loss(error)

                if self.train_rnn_w_ll:
                    l_loss = ll_loss(prediction_ll)

                    if self.weight_mse_by_ll is None:
                        srnn_loss = (1 - current_ll_weight) * p_loss + current_ll_weight * l_loss
                    else:
                        local_ll = torch.logsumexp(prediction_ll, dim=1)
                        local_ll = local_ll - local_ll.max()  # From 0 to -inf
                        local_ll = local_ll / local_ll.min()  # From 0 to 1 -> low LL is 1, high LL is 0: Inverse Het
                        local_ll = local_ll / local_ll.mean()  # Scale it to mean = 1

                        if self.weight_mse_by_ll == 'het':
                            # Het: low LL is 0, high LL is 1
                            local_ll = local_ll.max() - local_ll

                        srnn_loss = p_loss * (self.ll_weight - current_ll_weight) + \
                                    current_ll_weight * (error * local_ll).mean()
                else:
                    srnn_loss = p_loss
                    l_loss = 0

                if not self.srnn.config.use_cached_predictions:
                    srnn_loss.backward()

                    if self.srnn.config.clip_gradient_value > 0:
                        clip_grad_value_complex_(srnn_parameters, self.srnn.config.clip_gradient_value)

                    srnn_optimizer.step()

                if self.train_spn_on_prediction:
                    if hasattr(self.westimator, 'spn'):
                        westimator_loss = ll_loss_pred(prediction_ll, error.detach())

                        westimator_loss.backward()
                        westimator_optimizer.step()
                    else:
                        if type(prediction_ll) == tuple:
                            u, log_det = prediction_ll

                            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                            negloglik_loss += 0.5 * self.westimator.final_input_sizes * np.log(2 * pi)
                            negloglik_loss -= log_det
                            negloglik_loss = torch.mean(negloglik_loss * (error ** -2)) * 1e-4

                        else:
                            mu, logp = torch.chunk(prediction_ll, 2, dim=1)
                            u = (w_in - mu) * torch.exp(0.5 * logp)

                            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                            negloglik_loss += 0.5 * w_in.shape[1] * np.log(2 * pi)
                            negloglik_loss -= 0.5 * torch.sum(logp, dim=1)

                            negloglik_loss = torch.mean(negloglik_loss)

                        negloglik_loss.backward()
                        westimator_loss = negloglik_loss.item()
                        westimator_optimizer.step()

                    westimator_loss_e += westimator_loss.detach()

                l_loss = l_loss.detach() if not type(l_loss) == int else l_loss
                srnn_loss_p_e += p_loss.item()
                srnn_loss_ll_e += l_loss
                srnn_loss_e += srnn_loss.item()

                westimator_losses.append(westimator_loss.detach().cpu().numpy())
                srnn_losses.append(srnn_loss.detach().cpu().numpy())
                srnn_losses_p.append(p_loss.detach().cpu().numpy())
                srnn_losses_ll.append(l_loss)

                if (i + 1) % 10 == 0:
                    print(f'Epoch {epoch + 1} / {epochs}: Step {(i + 1)} / {len(idx_batches)}. '
                          f'Avg. WCSPN Loss: {westimator_loss_e / (i + 1)} '
                          f'Avg. SRNN Loss: {srnn_loss_e / (i + 1)}')

            lr_scheduler.step()

            if epoch < self.ll_weight_inc_dur and self.train_rnn_w_ll:
                if self.step_increase:
                    current_ll_weight = 0
                else:
                    current_ll_weight += ll_weight_increase
            elif self.train_rnn_w_ll:
                current_ll_weight = self.ll_weight

            westimator_loss_epoch = westimator_loss_e / len(idx_batches)
            srnn_loss_epoch = srnn_loss_e / len(idx_batches)
            print(f'Epoch {epoch + 1} / {epochs} done.'
                  f'Avg. WCSPN Loss: {westimator_loss_epoch} '
                  f'Avg. SRNN Loss: {srnn_loss_epoch}')

            print(f'Avg. SRNN-Prediction-Loss: {srnn_loss_p_e / len(idx_batches)}')
            print(f'Avg. SRNN-LL-Loss: {srnn_loss_ll_e / len(idx_batches)}')

            if len(westimator_losses_epoch) > 0 and not stop_cspn_training and \
                    not westimator_loss_epoch < westimator_losses_epoch[-1] - self.westimator_stop_threshold and not \
                    self.train_spn_on_prediction:
                westimator_patience_counter += 1

                print(f'Increasing patience counter to {westimator_patience_counter}')

                if westimator_patience_counter >= self.westimator_early_stopping:
                    stop_cspn_training = True
                    print('WCSPN training stopped!')

            else:
                westimator_patience_counter = 0
            rt.step()

            westimator_losses_epoch.append(westimator_loss_epoch)

            if False and epoch % 3 == 0:
                pred_val, _ = self.predict({key: x for key, x in val_x.items() if len(x) > 0}, mpe=False)
                self.srnn.net.train()
                self.westimator.spn.train()
                self.westimator.weight_nn.train()

                val_mse = np.mean([((p - val_y[key][:, :, -1]) ** 2).mean() for key, p in pred_val.items()])
                val_errors.append(val_mse)

        if not self.srnn.config.use_cached_predictions:
            predictions = []
            f_cs = []

            with torch.no_grad():
                print(x.shape[0] // batch_size + 1)
                for i in range(x.shape[0] // batch_size + 1):
                    batch_x, batch_y = x[i * batch_size:(i + 1) * batch_size], \
                                       y[i * batch_size:(i + 1) * batch_size]
                    if self.srnn.config.use_searched_srnn:
                        prediction, f_c = self.srnn.net(batch_x, batch_y, self.srnn.net.weights,
                                                        return_coefficients=True)
                    else:
                        prediction, f_c = self.srnn.net(batch_x, batch_y, return_coefficients=True)
                    #prediction, f_c = self.srnn.net(batch_x, batch_y, return_coefficients=True)

                    predictions.append(prediction)
                    f_cs.append(f_c)

            predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
            f_cs = torch.cat(f_cs, dim=0).detach().cpu().numpy()

            import pickle
            with open('srnn_train.pkl', 'wb') as f:
                pickle.dump((predictions, f_cs), f)

        #import matplotlib.pyplot as plt
        #plt.rcParams.update({'font.size': 48, 'figure.figsize': (60, 40)})
        #index = list(range(len(westimator_losses)))
        #plt.ylabel('LL')
        #plt.plot(index, westimator_losses, label='WCSPN-Loss (Negative LL)', color='blue')
        #plt.plot(index, srnn_losses_ll, label='SRNN-Loss (Negative LL)', color='green')
        #plt.legend(loc='upper right')
#
        #ax2 = plt.twinx()
        #ax2.set_ylabel('MSE', color='red')
        #ax2.plot(index, srnn_losses, label='SRNN-Loss Total', color='magenta')
        #ax2.plot(index, srnn_losses_p, label='SRNN-Loss Prediction', color='red')
        #ax2.legend(loc='upper left')
#
        #plt.savefig('res/plots/0_PWN_Training_losses')
#
        #plt.clf()
        #plt.plot(val_errors)
        #plt.savefig('res/plots/0_PWN_Val_MSE')
        #print(val_errors)
#
        #if self.train_rnn_w_ll:
        #    plt.clf()
        #    plt.plot(ll_weight_history)
        #    plt.ylabel('SRNN LL-Loss Weight (percentage of total loss)')
        #    plt.title('LL Weight Warmup')
        #    plt.savefig('res/plots/0_PWN_LLWeightWarmup')

    @torch.no_grad()
    def predict(self, x, batch_size=1024, pred_label='', mpe=False):

        predictions, f_c = self.srnn.predict(x, batch_size, pred_label=pred_label, return_coefficients=True)

        x_ = {key: x_[:, :, -1] for key, x_ in x.items()}

        f_c_ = {key: f.reshape((f.shape[0], -1)) for key, f in f_c.items()}

        if self.westimator.use_stft:
            ll = self.westimator.predict(x_, f_c_, stft_y=False, batch_size=batch_size)
        else:
            ll = self.westimator.predict(x_, predictions, batch_size=batch_size)

        if mpe:
            y_empty = {key: np.zeros((x[key].shape[0], self.srnn.net.amt_prediction_samples)) for key in x.keys()}
            predictions_mpe = self.westimator.predict_mpe({key: x_.copy() for key, x_ in x.items()},
                                                           y_empty, batch_size=batch_size)
            lls_mpe = self.westimator.predict(x_, {key: v[0] for key, v in predictions_mpe.items()},
                                              stft_y=False, batch_size=batch_size)

            # predictions, likelihoods, likelihoods_mpe, predictions_mpe
            return predictions, ll, lls_mpe, {key: v[1] for key, v in predictions_mpe.items()}

        else:
            return predictions, ll

    def save(self, filepath):
        self.srnn.save(filepath)
        self.westimator.save(filepath)

    def load(self, filepath):
        self.srnn.load(filepath)
        self.westimator.load(filepath)
        self.westimator.stft_module = self.srnn.net.stft

    def call_westimator(self, x, y):

        y_ = torch.stack([y.real, y.imag], dim=-1) if torch.is_complex(y) else y

        if hasattr(self.westimator, 'spn'):
            sum_params, leaf_params = self.westimator.weight_nn(x.reshape((x.shape[0], x.shape[1] *
                                                                           (2 if self.westimator.use_stft else 1),)))
            self.westimator.args.param_provider.sum_params = sum_params
            self.westimator.args.param_provider.leaf_params = leaf_params

            return self.westimator.spn(y_), y_
        else:
            val_in = torch.cat([x, y_], dim=1).reshape((x.shape[0], -1))
            return self.westimator.model(val_in), val_in
