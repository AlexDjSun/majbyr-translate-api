import os
import subprocess
import logging
import torch
import ttsmms.commons
import ttsmms.utils
from ttsmms import TextMapper
from ttsmms.models import SynthesizerTrn
from scipy.io.wavfile import write

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class TTS:
    def __init__(self, model_dir_path: str, uroman_dir:str=None) -> None:
        self.model_path = model_dir_path
        self.vocab_file = f"{self.model_path}/vocab.txt"
        self.config_file = f"{self.model_path}/config.json"
        self.uroman_dir = uroman_dir
        assert os.path.isfile(self.config_file), f"{self.config_file} doesn't exist"
        self.hps = ttsmms.utils.get_hparams_from_file(self.config_file)
        self.text_mapper = TextMapper(self.vocab_file)
        self.net_g = SynthesizerTrn(
            len(self.text_mapper.symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model
        )

        self.net_g.to(device)

        _ = self.net_g.eval()

        self.g_pth = f"{self.model_path}/G_100000.pth"
        _ = ttsmms.utils.load_checkpoint(self.g_pth, self.net_g, None)
        self.sampling_rate=self.hps.data.sampling_rate
        self.is_uroman = self.hps.data.training_files.split('.')[-1] == 'uroman'
    def _use_uroman(self, txt):
        if self.is_uroman != True:
            return txt
        if self.uroman_dir is None:
            tmp_dir = os.path.join(os.getcwd(),"uroman")
            if os.path.exists(tmp_dir) == False:
                cmd = f"git clone https://github.com/isi-nlp/uroman.git {tmp_dir}"
                logging.info(f"downloading uroman and save to {tmp_dir}")
                subprocess.check_output(cmd, shell=True)
            self.uroman_dir = tmp_dir
        uroman_pl = os.path.join(self.uroman_dir, "bin", "uroman.pl")
        logging.info("uromanize")
        txt =  self.text_mapper.uromanize(txt, uroman_pl)
        return txt
    def synthesis(self, txt, wav_path=None):
        txt = self._use_uroman(txt)
        txt = self.text_mapper.filter_oov(txt)
        stn_tst = self.text_mapper.get_text(txt, self.hps)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
            hyp = self.net_g.infer(
                x_tst, x_tst_lengths, noise_scale=.667,
                noise_scale_w=0.8, length_scale=1.0
            )[0][0,0].cpu().float().numpy()
        del x_tst, x_tst_lengths
        torch.cuda.empty_cache()
        
        if wav_path != None:
            write(wav_path, self.hps.data.sampling_rate, hyp)
            return wav_path
        return {"x":hyp,"sampling_rate":self.sampling_rate}
