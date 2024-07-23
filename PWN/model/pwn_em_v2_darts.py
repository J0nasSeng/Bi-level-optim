from .srnn import SpectralGRUNet
from .spectral_rnn.manifold_optimization import ManifoldOptimizer
from .spectral_rnn.cgRNN import clip_grad_value_complex_
from .transformer import TransformerPredictor, TransformerConfig, TransformerNet
from .wein.EinsumNetwork import EinsumNetwork
from .wein import WEinConfig, WEin
from .model import Model
from util.losses import SMAPE


import numpy as np
import torch
import torch.nn as nn
from rtpt import RTPT
#from darts.darts_cnn.model_search import Network as CWSPNModelSearch
#from darts.darts_rnn.model_search import RNNModelSearch

def load_arch_params(srnn=True):

    if srnn:
        arch_params_raw = np.array([[0.12425484508275986, 0.13287867605686188, 0.3053589165210724, 0.12407691776752472, 0.31343069672584534], [0.1471414864063263, 0.1716749519109726, 0.2689335346221924, 0.15429958701133728, 0.25795045495033264], [0.12622499465942383, 0.14418970048427582, 0.3033452332019806, 0.12716391682624817, 0.2990761697292328], [0.15721747279167175, 0.18280093371868134, 0.23875008523464203, 0.1699821650981903, 0.25124937295913696], [0.13004758954048157, 0.1471775621175766, 0.2947767972946167, 0.13212530314922333, 0.2958727777004242], [0.13210837543010712, 0.14104391634464264, 0.28938964009284973, 0.1323057860136032, 0.3051522374153137], [0.17168262600898743, 0.19226105511188507, 0.2182369828224182, 0.1866532266139984, 0.2311660647392273], [0.13781553506851196, 0.15585428476333618, 0.27699413895606995, 0.14252912998199463, 0.28680694103240967], [0.1359567940235138, 0.14785297214984894, 0.2819565534591675, 0.13781803846359253, 0.29641565680503845], [0.1345694661140442, 0.1378871351480484, 0.28920304775238037, 0.13495169579982758, 0.30338871479034424], [0.18426808714866638, 0.19968274235725403, 0.20449098944664001, 0.19579121470451355, 0.2157669961452484], [0.14991770684719086, 0.16388356685638428, 0.25402095913887024, 0.157046839594841, 0.2751309275627136], [0.14341603219509125, 0.1550014764070511, 0.2648983299732208, 0.1470896452665329, 0.28959453105926514], [0.13954532146453857, 0.14323531091213226, 0.27765604853630066, 0.1403735876083374, 0.2991897165775299], [0.1372540295124054, 0.13370491564273834, 0.2900788486003876, 0.13680413365364075, 0.30215808749198914], [0.19367198646068573, 0.2013566941022873, 0.20089995861053467, 0.19881699979305267, 0.20525440573692322], [0.16387887299060822, 0.17883221805095673, 0.23284240067005157, 0.1715090423822403, 0.2529374361038208], [0.15318745374679565, 0.16522406041622162, 0.2501991391181946, 0.15793684124946594, 0.2734524607658386], [0.14591865241527557, 0.1509062647819519, 0.2667388916015625, 0.14764773845672607, 0.28878843784332275], [0.1419358104467392, 0.13989904522895813, 0.28172406554222107, 0.14170022308826447, 0.29474085569381714], [0.13864924013614655, 0.13600076735019684, 0.28776228427886963, 0.13783687353134155, 0.2997508645057678], [0.1973322480916977, 0.20142269134521484, 0.20004023611545563, 0.19924011826515198, 0.20196473598480225], [0.1765821874141693, 0.18998417258262634, 0.2179604172706604, 0.18260882794857025, 0.2328644096851349], [0.16478556394577026, 0.17697773873806, 0.23308713734149933, 0.1698693037033081, 0.2552802562713623], [0.15487118065357208, 0.16157126426696777, 0.25039204955101013, 0.1575969159603119, 0.2755686640739441], [0.1478762924671173, 0.14858323335647583, 0.2695561647415161, 0.14874333143234253, 0.2852410078048706], [0.14325501024723053, 0.1419588327407837, 0.2781819999217987, 0.143024742603302, 0.29357942938804626], [0.13893824815750122, 0.13765373826026917, 0.28746873140335083, 0.13878203928470612, 0.2971571683883667], [0.19894593954086304, 0.20094114542007446, 0.1999133676290512, 0.1992943435907364, 0.20090517401695251], [0.18553516268730164, 0.19513802230358124, 0.21019703149795532, 0.18911023437976837, 0.22001954913139343], [0.17598596215248108, 0.18599717319011688, 0.21931026875972748, 0.17988651990890503, 0.23882010579109192], [0.1651717722415924, 0.1725427210330963, 0.23293372988700867, 0.16802926361560822, 0.2613224685192108], [0.15594720840454102, 0.15925058722496033, 0.25276708602905273, 0.1573585569858551, 0.2746766209602356], [0.14931944012641907, 0.14970822632312775, 0.2654818296432495, 0.1493559330701828, 0.2861345708370209], [0.14388667047023773, 0.1431637704372406, 0.27797552943229675, 0.14345373213291168, 0.29152026772499084], [0.13940495252609253, 0.13847973942756653, 0.28648674488067627, 0.13872390985488892, 0.29690462350845337]])
        arch_params = (arch_params_raw == arch_params_raw.max(axis=1, keepdims=1)).astype(float)
        arch_params_torch = torch.tensor(arch_params).cuda()

    else:
        arch_params_raw = 0
        arch_params = (arch_params_raw == arch_params_raw.max(axis=1, keepdims=1)).astype(float)
        arch_params_torch = torch.tensor(arch_params).cuda()

    return arch_params_torch

class PWNEM(Model):

    def __init__(self, hidden_size, output_size, fft_compression, window_size, overlap, device, c_config: WEinConfig, num_srnn_layers=2, train_spn_on_gt=True,
                 train_spn_on_prediction=False, train_rnn_w_ll=False, weight_mse_by_ll=None, always_detach=False,
                 westimator_early_stopping=5, step_increase=False, westimator_stop_threshold=.5,
                 westimator_final_learn=2, ll_weight=0.5, ll_weight_inc_dur=20, use_transformer=False,
                 smape_target=False):

        assert train_spn_on_gt or train_spn_on_prediction
        assert not train_rnn_w_ll or train_spn_on_gt

        if not use_transformer:
            self.srnn = SpectralGRUNet(hidden_size, output_size, device, num_srnn_layers, fft_compression, window_size, overlap).to(device) 
        else:
            #trans_cfg = TransformerConfig(normalize_fft=True, window_size=window_size,
            #                  fft_compression=fft_compression)
            trans_cfg = TransformerConfig(normalize_fft=True, window_size=window_size, dropout=0.1,
                  fft_compression=fft_compression, hidden_dim=64, embedding_dim=32,
                  num_enc_dec=2, is_complex=False, native_complex=False)

            trans_cfg.step_width = int(window_size * overlap)
            trans_cfg.value_dim = window_size // 2 + 1

            trans_cfg.compressed_value_dim = trans_cfg.value_dim // fft_compression
            trans_cfg.removed_freqs = trans_cfg.value_dim - trans_cfg.compressed_value_dim
            trans_cfg.input_dim = trans_cfg.compressed_value_dim
            self.srnn = TransformerNet(trans_cfg, trans_cfg.input_dim * 2, trans_cfg.hidden_dim,
                                  trans_cfg.input_dim * 2, trans_cfg.q, trans_cfg.k, trans_cfg.heads,
                                  trans_cfg.num_enc_dec, attention_size=trans_cfg.attention_size,
                                  dropout=trans_cfg.dropout, chunk_mode=trans_cfg.chunk_mode, pe=trans_cfg.pe,
                                  complex=trans_cfg.is_complex, native_complex=trans_cfg.native_complex, device=device).to(device)
        self.westimator = WEin(device, c_config)


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
        self.device = device

    def train(self, dataloader, epochs=70, lr=0.004, lr_decay=0.97):

        rtpt_obj = RTPT('JS', 'PWN_baseline', epochs)
        rtpt_obj.start()

        device = self.device

        self.westimator.stft_module = self.srnn.stft


        # Init westimator
        batch_x, batch_y = next(iter(dataloader))
        westimator_x_prototype, westimator_y_prototype = self.westimator.prepare_input(batch_x[:1, :].to(device), batch_y[:1, :].to(device)) # NOTE: different to original PWN code since we only use 1 feature
        self.westimator.config.input_size = westimator_x_prototype.shape[1] + westimator_y_prototype.shape[1]
        self.westimator.create_net()

        prediction_loss = lambda error: error.mean()
        ll_loss = lambda out: -1 * torch.logsumexp(out, dim=1).mean()
        if self.smape_target:
            p_base_loss = SMAPE()
        # MSE target
        else:
            p_base_loss = lambda out, label: nn.MSELoss(reduction='none')(out, label).mean(axis=1)

        srnn_parameters = list(self.srnn.parameters())

        if not self.use_transformer:
            srnn_optimizer = torch.optim.Adam(srnn_parameters, lr=0.001)
        else:
            srnn_optimizer = torch.optim.RMSprop(srnn_parameters, 0.0004)

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

        self.srnn.train()
        self.westimator.net.train()

        val_errors = []
        print(f'Starting Training of {self.identifier} model')
        for epoch in range(epochs):

            if self.train_rnn_w_ll:
                ll_weight_history.append(current_ll_weight)

            srnn_loss_p_e = 0
            srnn_loss_ll_e = 0

            srnn_loss_e = 0
            westimator_loss_e = 0
            for i, (batch_x, batch_y) in enumerate(dataloader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_x, batch_y = batch_x.to(torch.float32), batch_y.to(torch.float32)
                batch_westimator_x, batch_westimator_y = self.westimator.prepare_input(batch_x, batch_y)

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
                prediction_raw, f_c = self.srnn(batch_x, batch_y)

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

                srnn_loss.backward()
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
                    print(f'Epoch {epoch + 1} / {epochs}: Step {(i + 1)} / {len(dataloader)}. '
                          f'Avg. WCSPN Loss: {westimator_loss_e / (i + 1)} '
                          f'Avg. SRNN Loss: {srnn_loss_e / (i + 1)}')

            self.westimator.net.em_update()
            #lr_scheduler.step()

            rtpt_obj.step()

            if epoch < self.ll_weight_inc_dur and self.train_rnn_w_ll:
                if self.step_increase:
                    current_ll_weight = 0
                else:
                    current_ll_weight += ll_weight_increase
            elif self.train_rnn_w_ll:
                current_ll_weight = self.ll_weight

            westimator_loss_epoch = westimator_loss_e / len(dataloader)
            srnn_loss_epoch = srnn_loss_e / len(dataloader)
            print(f'Epoch {epoch + 1} / {epochs} done.'
                  f'Avg. WCSPN Loss: {westimator_loss_epoch} '
                  f'Avg. SRNN Loss: {srnn_loss_epoch}')

            print(f'Avg. SRNN-Prediction-Loss: {srnn_loss_p_e / len(dataloader)}')
            print(f'Avg. SRNN-LL-Loss: {srnn_loss_ll_e / len(dataloader)}')

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

    def prepareNets(self, dataloader, epochs=70, lr=0.004, lr_decay=0.97):

        #rtpt_obj = RTPT('JS', 'PWN_baseline', epochs)
        #rtpt_obj.start()

        device = self.device

        self.westimator.stft_module = self.srnn.stft


        # Init westimator
        batch_x, batch_y = next(iter(dataloader))
        westimator_x_prototype, westimator_y_prototype = self.westimator.prepare_input(batch_x[:1, :].to(device), batch_y[:1, :].to(device)) # NOTE: different to original PWN code since we only use 1 feature
        self.westimator.config.input_size = westimator_x_prototype.shape[1] + westimator_y_prototype.shape[1]
        self.westimator.create_net()

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
            y_empty = {key: np.zeros((x[key].shape[0], self.srnn.amt_prediction_samples)) for key in x.keys()}
            predictions_mpe = self.westimator.predict_mpe({key: x_.copy() for key, x_ in x.items()},
                                                           y_empty, batch_size=batch_size)
            lls_mpe = self.westimator.predict(x_, {key: v[0] for key, v in predictions_mpe.items()},
                                              stft_y=False, batch_size=batch_size)

            return predictions, ll, lls_mpe, {key: v[1] for key, v in predictions_mpe.items()}

        else:
            return predictions, ll

    @torch.no_grad()
    def predict_v2(self, dataloader):
        preds, lls, gt = [], [], []
        for x, y in dataloader:
            x, y = x.to(torch.float32), y.to(torch.float32)
            x, y = x.to(self.device), y.to(self.device)
            predictions, f_c = self.srnn(x, y)
            batch_westimator_x, batch_westimator_y = self.westimator.prepare_input(x, y)
            prediction_ll, w_in = self.call_westimator(batch_westimator_x, f_c.reshape(f_c.shape[0], -1))
            preds.append(predictions)
            lls.append(prediction_ll)
            gt.append(y)
        return torch.cat(preds), torch.cat(lls), torch.cat(gt)

    def save(self, filepath):
        self.srnn.save(filepath)
        self.westimator.save(filepath)

    def load(self, filepath):
        self.srnn.load(filepath)
        self.westimator.load(filepath)
        self.westimator.stft_module = self.srnn.stft

    def call_westimator(self, x, y, return_cond=False):
        y_ = torch.stack([y.real, y.imag], dim=-1) if torch.is_complex(y) else y

        val_in = torch.cat([x, y_], dim=1)
        ll_joint = self.westimator.net(val_in)

        if return_cond:
            self.westimator.net.set_marginalization_idx(list(range(x.shape[1], x.shape[1] + y_.shape[1])))
            ll_marginal = self.westimator.net(val_in)
            self.westimator.net.set_marginalization_idx([])

            ll_cond = ll_joint - ll_marginal
            return ll_joint, ll_marginal, ll_cond, val_in

        else:
            return ll_joint, val_in