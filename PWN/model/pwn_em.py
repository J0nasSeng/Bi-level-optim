from .spectral_rnn.spectral_rnn import SpectralRNN, SpectralRNNConfig
from .spectral_rnn.manifold_optimization import ManifoldOptimizer
from .spectral_rnn.cgRNN import clip_grad_value_complex_
from .transformer import TransformerPredictor, TransformerConfig
from .wein.EinsumNetwork import EinsumNetwork
from .wein import WEinConfig, WEin
from .model import Model

import numpy as np
import torch
import torch.nn as nn
#from darts.darts_cnn.model_search import Network as CWSPNModelSearch
from darts.darts_rnn.model_search import RNNModelSearch

# Use GPU if avaiable
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

def load_arch_params(srnn=True):

    if srnn:
        arch_params_raw = np.array([[0.050721749663352966, 0.04433192312717438, 0.16527917981147766, 0.18307217955589294, 0.5565950274467468], [0.040467411279678345, 0.07901535928249359, 0.18646566569805145, 0.05171829089522362, 0.6423332095146179], [0.04263773560523987, 0.08615807443857193, 0.2711254060268402, 0.06114431843161583, 0.5389344692230225], [0.07516927272081375, 0.12268523871898651, 0.096587173640728, 0.4311600625514984, 0.27439823746681213], [0.08796003460884094, 0.15701040625572205, 0.12808452546596527, 0.3574199080467224, 0.26952511072158813], [0.04176683351397514, 0.06611082702875137, 0.24522650241851807, 0.043979208916425705, 0.6029165983200073], [0.12114150822162628, 0.12562872469425201, 0.12486584484577179, 0.4862768352031708, 0.14208708703517914], [0.13396839797496796, 0.151680126786232, 0.14670224487781525, 0.3892991840839386, 0.17835000157356262], [0.04916372522711754, 0.13553066551685333, 0.1763349175453186, 0.05549423024058342, 0.5834764838218689], [0.04685823246836662, 0.07003746926784515, 0.2816809117794037, 0.04922410845756531, 0.5521993041038513], [0.15368123352527618, 0.14700733125209808, 0.15072952210903168, 0.4021605849266052, 0.1464213728904724], [0.15976585447788239, 0.1600133776664734, 0.16294682025909424, 0.3517597019672394, 0.16551423072814941], [0.058689650148153305, 0.1703670173883438, 0.16724269092082977, 0.07291232794523239, 0.5307883024215698], [0.055038899183273315, 0.09913697838783264, 0.2591077983379364, 0.06165001913905144, 0.5250662565231323], [0.046971675008535385, 0.05606105178594589, 0.3368713855743408, 0.04834522679448128, 0.5117506980895996], [0.1700579822063446, 0.20108993351459503, 0.16932512819766998, 0.2598014771938324, 0.1997254639863968], [0.16863587498664856, 0.201965793967247, 0.17416375875473022, 0.25257205963134766, 0.20266251266002655], [0.07215452939271927, 0.17561109364032745, 0.16792818903923035, 0.09066563844680786, 0.4936405122280121], [0.06556905806064606, 0.11974447220563889, 0.23116466403007507, 0.07662562280893326, 0.5068961977958679], [0.05449848249554634, 0.07043719291687012, 0.30413123965263367, 0.05775744467973709, 0.5131755471229553], [0.046172287315130234, 0.04911089688539505, 0.3488101065158844, 0.046330392360687256, 0.5095762610435486], [0.17062576115131378, 0.2239842563867569, 0.1838742047548294, 0.19763751327991486, 0.22387832403182983], [0.1700160950422287, 0.2217603176832199, 0.18901807069778442, 0.19648462533950806, 0.22272087633609772], [0.09145978093147278, 0.18254666030406952, 0.18409258127212524, 0.1088210865855217, 0.43307989835739136], [0.0799390971660614, 0.1332632303237915, 0.22614923119544983, 0.09275742620229721, 0.46789100766181946], [0.06428228318691254, 0.08622068166732788, 0.278938889503479, 0.06966854631900787, 0.5008896589279175], [0.05281244218349457, 0.05821657180786133, 0.3251934349536896, 0.053853970021009445, 0.5099236369132996], [0.04454782232642174, 0.04546264559030533, 0.3560258150100708, 0.04416605457663536, 0.5097976326942444], [0.1830390989780426, 0.21636149287223816, 0.1949465125799179, 0.18895216286182404, 0.2167007178068161], [0.18284614384174347, 0.21496586501598358, 0.1979825496673584, 0.18857164680957794, 0.21563377976417542], [0.1232658252120018, 0.2102123200893402, 0.2096281349658966, 0.13761258125305176, 0.31928113102912903], [0.10835397988557816, 0.16454021632671356, 0.2349458932876587, 0.12079676985740662, 0.37136316299438477], [0.08393312990665436, 0.1134614497423172, 0.25494706630706787, 0.09207645058631897, 0.455581933259964], [0.0655510276556015, 0.07645338773727417, 0.29303377866744995, 0.06834102421998978, 0.49662071466445923], [0.05263996496796608, 0.055064380168914795, 0.32972007989883423, 0.05274389311671257, 0.5098316669464111], [0.04318319261074066, 0.04349406808614731, 0.36913397908210754, 0.04275780916213989, 0.5014309883117676]])
        arch_params = (arch_params_raw == arch_params_raw.max(axis=1, keepdims=1)).astype(float)
        arch_params_torch = torch.tensor(arch_params).to(device=device)

    else:
        arch_params_raw = 0
        arch_params = (arch_params_raw == arch_params_raw.max(axis=1, keepdims=1)).astype(float)
        arch_params_torch = torch.tensor(arch_params).to(device=device)

    return arch_params_torch

class PWNEM(Model):

    def __init__(self, s_config: SpectralRNNConfig, c_config: WEinConfig, train_spn_on_gt=True,
                 train_spn_on_prediction=False, train_rnn_w_ll=False, weight_mse_by_ll=None, always_detach=False,
                 westimator_early_stopping=5, step_increase=False, westimator_stop_threshold=.5,
                 westimator_final_learn=2, ll_weight=0.5, ll_weight_inc_dur=20, use_transformer=False,
                 smape_target=False):

        assert train_spn_on_gt or train_spn_on_prediction
        assert not train_rnn_w_ll or train_spn_on_gt

        self.srnn = SpectralRNN(s_config) if not use_transformer else TransformerPredictor(
            TransformerConfig(normalize_fft=True, window_size=s_config.window_size,
                              fft_compression=s_config.fft_compression))
        self.westimator = WEin(c_config)


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
        self.smape_target = smape_target

    def train(self, x_in, y_in, val_x, val_y, embedding_sizes, batch_size=256, epochs=100, lr=0.004, lr_decay=0.97):

        # TODO: Adjustment for complex optimization, needs documentation
        if type(self.srnn) == TransformerPredictor:
            lr /= 10
        elif False and self.srnn.config.rnn_layer_config.use_cg_cell:
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
        self.westimator.config.input_size = westimator_x_prototype.shape[1] + westimator_y_prototype.shape[1]
        self.westimator.create_net()

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
        amt_param = sum([p.numel() for p in self.srnn.net.parameters()])
        amt_param_w = sum([p.numel() for p in self.westimator.parameters()])

        srnn_optimizer = ManifoldOptimizer(srnn_parameters, lr, torch.optim.RMSprop, alpha=0.9) \
            if type(self.srnn.config) == SpectralRNNConfig and self.srnn.config.rnn_layer_config.use_cg_cell \
            else torch.optim.RMSprop(srnn_parameters, lr=lr, alpha=0.9)

        if self.train_rnn_w_ll:
            current_ll_weight = 0
            ll_weight_history = []
            ll_weight_increase = self.ll_weight / self.ll_weight_inc_dur
        elif self.train_spn_on_prediction:
            def ll_loss_pred(out, error):
                return (-1 * torch.logsumexp(out, dim=1) * (error ** -2)).sum() * 1e-4

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=srnn_optimizer, gamma=lr_decay)

        westimator_losses = []
        srnn_losses = []
        srnn_losses_p = []
        srnn_losses_ll = []

        stop_cspn_training = False
        westimator_patience_counter = 0
        westimator_losses_epoch = []

        self.srnn.net.train()
        self.westimator.net.train()

        if self.srnn.config.use_cached_predictions:
            import pickle
            with open('srnn_train.pkl', 'rb') as f:
                all_predictions, all_f_cs = pickle.load(f)

                all_predictions = torch.from_numpy(all_predictions).to(device)
                all_f_cs = torch.from_numpy(all_f_cs).to(device)

        val_errors = []
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

            srnn_loss_e = 0
            westimator_loss_e = 0
            for i, idx in enumerate(idx_batches):
                batch_x, batch_y = x[idx].detach().clone(), y[idx, :].detach().clone()
                batch_westimator_x, batch_westimator_y = self.westimator.prepare_input(batch_x[:, :, -1], batch_y)

                if self.srnn.config.use_cached_predictions:
                    batch_p = all_predictions.detach().clone()[idx]
                    batch_fc = all_f_cs.detach().clone()[idx]

                if self.train_spn_on_gt:
                    if not stop_cspn_training or epoch >= epochs - self.westimator_final_learn:
                        out_w, _ = self.call_westimator(batch_westimator_x, batch_westimator_y)

                        gt_ll = EinsumNetwork.log_likelihoods(out_w)
                        westimator_loss = gt_ll.sum()
                        westimator_loss.backward()

                        self.westimator.net.em_process_batch()

                    else:
                        westimator_loss = westimator_losses_epoch[-1]

                    westimator_loss_e += westimator_loss.detach()

                srnn_optimizer.zero_grad()

                if not self.srnn.config.use_cached_predictions:
                    if self.srnn.config.use_searched_srnn:
                        prediction_raw, f_c = self.srnn.net(batch_x, batch_y,self.srnn.net.weights, return_coefficients=True)
                    else:
                        prediction_raw, f_c = self.srnn.net(batch_x, batch_y, return_coefficients=True)
                else:
                    prediction_raw, f_c = batch_p, batch_fc

                if self.westimator.use_stft:
                    prediction_ll_ = self.call_westimator(batch_westimator_x,
                                                          f_c.reshape((f_c.shape[0], -1))
                                                          if self.train_rnn_w_ll and not self.always_detach
                                                          else f_c.reshape((f_c.shape[0], -1)).detach(),
                                                          return_cond=self.train_rnn_w_ll)
                else:
                    prediction_ll_ = self.call_westimator(batch_westimator_x,
                                                          prediction if self.train_rnn_w_ll and not self.always_detach
                                                          else prediction.detach(),
                                                          return_cond=self.train_rnn_w_ll)

                if self.train_rnn_w_ll:
                    prediction_ll, _, prediction_ll_cond, w_win = prediction_ll_
                else:
                    prediction_ll, _ = prediction_ll_

                if self.westimator.use_stft:
                    prediction_ll, w_in = self.call_westimator(batch_westimator_x, f_c.reshape((f_c.shape[0], -1))
                                        if self.train_rnn_w_ll and not self.always_detach else f_c.reshape((f_c.shape[0], -1)).detach())
                else:
                    prediction_ll, w_in = self.call_westimator(batch_westimator_x, prediction
                                        if self.train_rnn_w_ll and not self.always_detach else prediction.detach())

                if type(self.srnn) == TransformerPredictor:
                    prediction = prediction_raw
                else:
                    prediction = prediction_raw

                error = p_base_loss(prediction, batch_y)
                p_loss = prediction_loss(error)

                prediction_ll = EinsumNetwork.log_likelihoods(prediction_ll)
                if self.train_rnn_w_ll:
                    l_loss = ll_loss(prediction_ll)

                    if self.weight_mse_by_ll is None:
                        srnn_loss = (1 - current_ll_weight) * p_loss + current_ll_weight * prediction_ll_cond[:, 0]
                    else:
                        local_ll = prediction_ll_cond[:, 0]
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
                    westimator_loss = ll_loss_pred(prediction_ll, error.detach())
                    westimator_loss.backward()

                    self.westimator.net.em_process_batch()

                    westimator_loss_e += westimator_loss.detach()

                l_loss = l_loss.detach() if not type(l_loss) == int else l_loss
                srnn_loss_p_e += p_loss.detach()
                srnn_loss_ll_e += l_loss
                srnn_loss_e += srnn_loss.detach()

                westimator_losses.append(westimator_loss.detach().cpu().numpy())#westimator_losses.append(westimator_loss.detach())
                srnn_losses.append(srnn_loss.detach().cpu().numpy())#srnn_losses.append(srnn_loss.detach())
                srnn_losses_p.append(p_loss.detach().cpu().numpy())#srnn_losses_p.append(p_loss.detach())
                srnn_losses_ll.append(l_loss)#srnn_losses_ll.append(l_loss)

                if (i + 1) % 10 == 0:
                    print(f'Epoch {epoch + 1} / {epochs}: Step {(i + 1)} / {len(idx_batches)}. '
                          f'Avg. WCSPN Loss: {westimator_loss_e / (i + 1)} '
                          f'Avg. SRNN Loss: {srnn_loss_e / (i + 1)}')

            self.westimator.net.em_update()
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
                    not westimator_loss_epoch > westimator_losses_epoch[-1] + self.westimator_stop_threshold and not \
                    self.train_spn_on_prediction:
                westimator_patience_counter += 1

                print(f'Increasing patience counter to {westimator_patience_counter}')

                if westimator_patience_counter >= self.westimator_early_stopping:
                    stop_cspn_training = True
                    print('WEIN training stopped!')

            else:
                westimator_patience_counter = 0

            westimator_losses_epoch.append(westimator_loss_epoch)

            if False and epoch % 3 == 0:
                pred_val, _ = self.predict({key: x for key, x in val_x.items() if len(x) > 0}, mpe=False)
                self.srnn.net.train()
                self.westimator.net.train()

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
                    prediction, f_c = self.srnn.net(batch_x, batch_y, return_coefficients=True)

                    predictions.append(prediction)
                    f_cs.append(f_c)

            predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
            f_cs = torch.cat(f_cs, dim=0).detach().cpu().numpy()

            import pickle
            with open('srnn_train.pkl', 'wb') as f:
                pickle.dump((predictions, f_cs), f)

        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 48, 'figure.figsize': (60, 40)})
        index = list(range(len(westimator_losses)))
        plt.ylabel('LL')
        plt.plot(index, westimator_losses, label='WCSPN-Loss (Negative LL)', color='blue')
        plt.plot(index, srnn_losses_ll, label='SRNN-Loss (Negative LL)', color='green')
        plt.legend(loc='upper right')

        ax2 = plt.twinx()
        ax2.set_ylabel('MSE', color='red')
        ax2.plot(index, srnn_losses, label='SRNN-Loss Total', color='magenta')
        ax2.plot(index, srnn_losses_p, label='SRNN-Loss Prediction', color='red')
        ax2.legend(loc='upper left')

        plt.savefig('res/plots/0_PWN_Training_losses')

        plt.clf()
        plt.plot(val_errors)
        plt.savefig('res/plots/0_PWN_Val_MSE')
        print(val_errors)

        if self.train_rnn_w_ll:
            plt.clf()
            plt.plot(ll_weight_history)
            plt.ylabel('SRNN LL-Loss Weight (percentage of total loss)')
            plt.title('LL Weight Warmup')
            plt.savefig('res/plots/0_PWN_LLWeightWarmup')

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

    def call_westimator(self, x, y, return_cond=False):
        y_ = torch.stack([y.real, y.imag], dim=-1) if torch.is_complex(y) else y

        val_in = torch.cat([x, y_], dim=1).to(device=device)
        ll_joint = self.westimator.net(val_in)

        if return_cond:
            self.westimator.net.set_marginalization_idx(list(range(x.shape[1], x.shape[1] + y_.shape[1])))
            ll_marginal = self.westimator.net(val_in)
            self.westimator.net.set_marginalization_idx([])

            ll_cond = ll_joint - ll_marginal
            return ll_joint, ll_marginal, ll_cond, val_in

        else:
            return ll_joint, val_in
